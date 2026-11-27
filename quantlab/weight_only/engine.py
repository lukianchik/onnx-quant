"""Квантование весов, проверка выходов и экспорт модели с отчётом."""

import json
import math
import time
from pathlib import Path

import numpy as np
import onnx

from quantlab.core import ModelError, load_model, run_directory
from quantlab.runtime import session
from quantlab.weight_only.graph import combine_groups, discover, export
from quantlab.weight_only.minmax import minmax


def _verify_outputs(reference, candidate, feeds):
    """Проверить формы, типы и конечность выходов; измерить MSE без порога качества."""
    specs = [(v.name, v.type) for v in reference.get_outputs()]
    if specs != [(v.name, v.type) for v in candidate.get_outputs()]:
        raise ModelError('Изменились имена или типы выходов')
    sums = {name: 0.0 for name, _ in specs}
    counts = dict.fromkeys(sums, 0)
    runs = 0
    for feed in feeds:
        ref = reference.run(None, feed)
        got = candidate.run(None, feed)
        for (name, _), a, b in zip(specs, ref, got, strict=True):
            if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
                raise ModelError(f'Неподдерживаемый тип выхода: {name}')
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ModelError(f'Изменилась форма или тип выхода: {name}')
            if a.dtype.kind in 'fiu':
                if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ModelError(f'Пустой или неконечный выход: {name}')
                diff = a.astype(np.float64) - b.astype(np.float64)
                with np.errstate(over='ignore', invalid='ignore'):
                    err = float(np.sum(diff**2))
                if not np.isfinite(err + sums[name]):
                    raise ModelError(f'Переполнение MSE: {name}')
                sums[name] += err
                counts[name] += a.size
            elif not np.array_equal(a, b):
                raise ModelError(f'Изменился нечисловой выход: {name}')
        runs += 1
    return {
        'status': 'measured' if runs else 'not_run_no_inputs',
        'batches': runs,
        'mse_by_output': {n: sums[n] / c for n, c in counts.items() if c},
    }


def _model_metrics(path, output, model, layers, skipped):
    """Собрать общие метрики сохранённой и проверенной модели."""
    fp32 = sum(math.prod(t.dims) for t in model.graph.initializer if t.data_type == onnx.TensorProto.FLOAT)
    fp16 = sum(math.prod(t.dims) for t in model.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16)
    total = fp32 + fp16
    half_layers = sum(layer.get('source_dtype') == 'float16' for layer in layers)
    covered = sum(layer['parameters'] for layer in layers)
    before, after = path.stat().st_size, output.stat().st_size
    return {
        'representation': 'weight-only INT8 + DQ', 'provider': 'CPUExecutionProvider',
        'selected_count': len(layers), 'selected_nodes': [layer['node'] for layer in layers],
        'layers': layers, 'skipped': skipped, 'quantized_parameters': covered,
        'fp32_initializer_parameters': fp32, 'fp16_initializer_parameters': fp16,
        'floating_initializer_parameters': total,
        'coverage_percent': 100 * covered / total if total else 0.,
        'source_size_mib': before / 1024**2,
        'fp32_size_mib': before / 1024**2,  # Совместимость со старой JSON-схемой.
        'int8_size_mib': after / 1024**2,
        'size_reduction_percent': 100 * (1 - after / before),
        'quantized_operators': {'DequantizeLinear': len(layers), **({'Cast': half_layers} if half_layers else {})},
        'onnx_valid': True, 'onnxruntime_load_ok': True,
    }


def quantize_standard(path, feeds=()):
    """Квантовать поддерживаемые FP32/FP16-веса MinMax, не изменяя исходную модель."""
    started = time.perf_counter()
    path = Path(path)
    model = load_model(path)
    onnx.checker.check_model(model)
    targets, skipped = discover(model)
    if not targets:
        raise ModelError(f'Нет поддерживаемых постоянных FP32/FP16-весов: {skipped}')
    bank = {}
    details = []
    for target in targets:
        pairs = [minmax(w) for w in target.matrices()]
        bank[target.weight] = combine_groups(pairs)
        details.append({'node': target.node, 'operator': target.op,
                        'weight': target.weight, 'axis': target.axis, 'source_dtype': str(target.original.dtype),
                        'parameters': int(target.original.size), 'method': 'MinMax'})
    with run_directory(path.parent, 'standard_') as run:
        output = run / 'quantized_standard.onnx'
        onnx.save(export(model, targets, bank), str(output))
        onnx.checker.check_model(str(output))
        ref_session = session(path)
        int8_session = session(output)
        fidelity = _verify_outputs(ref_session, int8_session, feeds)
        result = _model_metrics(path, output, model, details, skipped)
        result.update({
            'mode': 'standard', 'algorithm': 'MinMax',
            'output_fidelity': fidelity, 'adapter_activation_validation': 'not_run_standard',
            'elapsed_seconds': time.perf_counter() - started,
        })
        (run / 'report.json').write_text(json.dumps({'result': result}, ensure_ascii=False,
            indent=2, allow_nan=False), encoding='utf-8')
        return output, result


def quantize_fast(path, feeds):
    """Запустить неизменённую политику Single-Robust."""
    return _quantize_adaptive(path, feeds, "fast")


def quantize_balanced(path, feeds, work_limit=None):
    """Запустить Progressive с ограниченным бюджетом поиска."""
    return _quantize_adaptive(path, feeds, "balanced", work_limit)


def quantize_quality(path, feeds, work_limit=None, task='generic'):
    """Запустить Progressive-Global, с явным типом задачи для V2."""
    return _quantize_adaptive(path, feeds, 'quality', work_limit, task)


def _quantize_adaptive(path, feeds, mode, work_limit=None, task='generic'):
    """Использовать общий экспорт и Global Guard для адаптивных политик."""
    from quantlab.weight_only.calibration import capture, activation_info
    from quantlab.weight_only.fast import split_feeds, select_single, guard_decision, TOTAL_WORK
    from quantlab.weight_only.progressive import select_progressive, BUDGETS
    from quantlab.weight_only.progressive import METHODS
    from quantlab.weight_only.global_search import CandidatePool, split_quality, search, check_score
    quality = mode == 'quality'
    if task not in ('generic', 'classification'):
        raise ModelError('Недопустимый тип задачи')
    algorithm = {'quality':'Progressive-Global','balanced':'Progressive','fast':'Single-Robust'}[mode]
    selector = select_single if mode == 'fast' else select_progressive
    pool = CandidatePool() if quality else None
    local_selection = {}
    limit = TOTAL_WORK if work_limit is None else work_limit
    if type(limit) is not int or limit not in BUDGETS:
        raise ModelError('Недопустимый бюджет поиска')
    started = time.perf_counter()
    path = Path(path)
    if quality:
        fit, dev, search_feeds, check = split_quality(list(feeds))
    else:
        fit, dev, check = split_feeds(list(feeds))
    model = load_model(path)
    onnx.checker.check_model(model)
    targets, skipped = discover(model)
    base, proposal, details, valid = {}, {}, [], []
    info = activation_info(model, targets)
    work = 0
    for i, target in enumerate(targets):
        # Ограничение пула: не выделять чрезмерную память под широкие слои.
        matrices = target.matrices()
        if sum(w.shape[0] for w in matrices)*512*4 > 32*1024**2:
            skipped.append({'node': target.node, 'reason': 'activation_memory_budget'})
            continue
        try:
            x,d = capture(model, target, matrices, info, fit, dev, session, 42 + i * 101)
        except Exception as exc:
            skipped.append({'node':target.node,'reason':str(exc)[:400]})
            continue
        try:
            if quality:
                variants, losses = {}, {}
                def retain(name, pairs, loss):
                    """Удержать построенный и оценённый вариант текущего оператора."""
                    variants[name] = combine_groups(pairs)
                    losses[name] = loss
                b,p,cost,diag = selector(matrices,x,d,limit-work,retain=retain)
                local_name = next(k for k,v in METHODS.items() if v == diag['method'])
                pool.add(target.weight, variants, losses, local_name)
                local_selection[target.weight] = local_name
                variants.clear()
                losses.clear()
            else:
                b,p,cost,diag = selector(matrices,x,d,limit-work)
        except (ValueError, ArithmeticError) as exc:
            skipped.append({'node':target.node,'reason':str(exc)[:400]})
            continue
        work += cost
        base[target.weight] = combine_groups(b)
        proposal[target.weight] = base[target.weight] if p is b else combine_groups(p)
        valid.append(target)
        details.append({'node':target.node,'weight':target.weight,'operator':target.op,
                        'axis':target.axis,'source_dtype':str(target.original.dtype),'parameters':int(target.original.size),**diag})
    if not valid:
        raise ModelError(f'Нет проверенных адаптеров {mode}: {skipped}')
    with run_directory(path.parent, mode + '_') as run:
        output = run / f'quantized_{mode}.onnx'
        baseline = run / 'minmax_reference.onnx'
        onnx.save(export(model,valid,base),str(baseline))
        reference = session(path)
        old = _verify_outputs(reference,session(baseline),check)
        new = None
        accepted = False
        global_info = {'kind':'none','calls':0,'cap':0,'active':[]}
        global_checks = []
        final_selection = None
        reason = 'no_local_gain'
        if quality:
            options = pool.options()
            def bank_for(selection):
                """Подставить альтернативы только для активных операторов."""
                return {**proposal, **{k: options[k][selection[k]] for k in options}}
            def model_session(bank):
                """Создать сессию из экспортированного графа без временных файлов."""
                return session(export(model,valid,bank).SerializeToString())
            def evaluate(selection):
                """Измерить все выходы на Search относительно исходной модели."""
                return _verify_outputs(reference,model_session(bank_for(selection)),search_feeds)
            def infer(weights, split):
                """V2 получает логиты всех пакетов, не трогая Check."""
                session = reference if weights is None else model_session({**proposal, **weights})
                if len(session.get_outputs()) != 1:
                    raise ValueError('V2 требует единственный выход логитов')
                batches = fit if split == 'fit' else search_feeds
                rows = [session.run(None, feed)[0] for feed in batches]
                if any(not isinstance(z,np.ndarray) or z.ndim != 2 or z.shape[1] < 2 for z in rows):
                    raise ValueError('V2 требует логиты [batch, classes]')
                return np.concatenate(rows,axis=0)
            t0 = time.perf_counter()
            global_selection, global_info = search(local_selection,options,evaluate,
                classification=task=='classification',infer=infer)
            global_info.update(seconds=time.perf_counter()-t0,bank_bytes=pool.bytes,
                               bank_limit_bytes=pool.cap,dropped_banks=pool.dropped,task=task)
            best_score = None
            best_model = None
            reason = 'no_global_gain'
            trials = [('local',local_selection)]
            if global_selection != local_selection:
                trials.append(('global',global_selection))
            for name,selection in trials:
                try:
                    candidate_model = export(model,valid,bank_for(selection))
                    measured = _verify_outputs(reference,session(candidate_model.SerializeToString()),check)
                    score = check_score(old,measured)
                    global_checks.append({'selection':name,'accepted':score is not None,
                                          'score':score,'fidelity':measured})
                    if score is not None and (best_score is None or score < best_score):
                        best_model = candidate_model
                        best_score,new,final_selection = score,measured,selection
                        accepted,reason = True,'accepted_'+name
                except Exception as exc:
                    global_checks.append({'selection':name,'accepted':False,'error':str(exc)[:300]})
            if accepted:
                onnx.save(best_model,str(output))
            for d in details:
                d['local_method'] = d['method']
                d['method'] = METHODS[final_selection[d['weight']]] if accepted else 'MinMax'
        elif any(d['method']!='MinMax' for d in details):
            try:
                onnx.save(export(model,valid,proposal),str(output))
                new = _verify_outputs(reference,session(output),check)
                accepted = guard_decision(old,new)
                reason = 'accepted' if accepted else 'no_global_gain'
            except Exception as exc:
                reason = 'proposal_invalid: ' + str(exc)[:300]
        if not accepted:
            baseline.replace(output)
            if not quality:
                for d in details:
                    d['local_method'] = d['method']
                    d['method'] = 'MinMax'
        onnx.checker.check_model(str(output))
        result = _model_metrics(path, output, model, details, skipped)
        result.update({'mode':mode,'algorithm':algorithm,'effective_algorithm':algorithm if accepted else 'MinMax',
                  'output_fidelity':new if accepted else old,'adapter_activation_validation':'validated_targets_only',
                  'global_guard':{'accepted':accepted,'reason':reason,'baseline':old,'proposal':new,
                                  'baseline_scope':'same_validated_targets'},
                  'split_batches':{'fit':len(fit),'dev':len(dev),'check':len(check)},
                  'work_budget':{'used':work,'limit':limit},
                  'elapsed_seconds':time.perf_counter()-started})
        if quality:
            result['split_batches']['search'] = len(search_feeds)
            result['global_search'] = global_info
            result['global_checks'] = global_checks
            result['local_selection'] = local_selection
            result['selection'] = final_selection if accepted else dict.fromkeys(local_selection,'minmax')
            result['effective_algorithm'] = ('Progressive-Global' if reason == 'accepted_global'
                else 'Progressive' if accepted else 'MinMax')
        baseline.unlink(missing_ok=True)
        (run/'report.json').write_text(json.dumps({'result':result},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return output,result
