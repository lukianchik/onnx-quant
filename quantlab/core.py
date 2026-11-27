"""Анализ ONNX-графа и осторожный выбор узлов для квантования."""

from collections import Counter
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid

import onnx
from onnx import TensorProto

from quantlab.weight_only.graph import iter_targets, supports_weight_only

MAX_MODEL_BYTES = 256 * 1024 * 1024


class ModelError(ValueError):
    """Ошибка неподдерживаемой модели или входных данных."""


def reject_external_data(message):
    """Отклонить внешние тензоры во всех вложенных protobuf-полях."""
    if isinstance(message, TensorProto):
        if message.data_location == TensorProto.EXTERNAL or message.external_data:
            raise ModelError("Внешние файлы весов пока не поддерживаются")
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            if (field.is_repeated if hasattr(field, "is_repeated")
                    else field.label == field.LABEL_REPEATED):
                for item in value:
                    reject_external_data(item)
            else:
                reject_external_data(value)


def load_model(path: Path):
    """Загрузить автономную модель допустимого размера до обращения к ORT."""
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MODEL_BYTES:
        raise ModelError('Ожидается ONNX-файл размером до 256 МиБ')
    model = onnx.load(path, load_external_data=False)
    reject_external_data(model)
    return model


@contextmanager
def run_directory(parent: Path, prefix: str = ''):
    """Выделить уникальный каталог и убрать неполный результат при ошибке."""
    directory = parent / (prefix + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def stage_model(source: str | Path, run_root: Path) -> Path:
    """Скопировать и проверить ONNX-модель в отдельном каталоге запуска."""
    source = Path(source)
    if source.suffix.lower() != '.onnx' or not source.is_file():
        raise ModelError('Ожидается существующий файл .onnx')
    if not (0 < source.stat().st_size <= MAX_MODEL_BYTES):
        raise ModelError('Файл пустой или превышает лимит 256 МиБ')

    with run_directory(run_root) as run_dir:
        original = run_dir / 'source.onnx'
        shutil.copyfile(source, original)
        model = load_model(original)
        used = set()
        for index, node in enumerate(model.graph.node):
            if not node.name or node.name in used:
                name = f'quantlab_{node.op_type}_{index}'
                while name in used:
                    name += '_'
                node.name = name
            used.add(node.name)
        prepared = run_dir / 'prepared.onnx'
        onnx.save(model, str(prepared))
        onnx.checker.check_model(str(prepared))
        return prepared


def analyze(path: str | Path) -> dict:
    """Вернуть сведения о графе и поддерживаемых линейных операторах в формате JSON."""
    path = Path(path)
    model = load_model(path)
    counts = Counter(n.op_type for n in model.graph.node)
    skipped = []
    candidates = []
    if supports_weight_only(model):
        for target in iter_targets(model, skipped):
            shape = list(target.original.shape)
            candidates.append({
                'node': target.node, 'operator': target.op, 'weight': target.weight,
                'axis': target.axis, 'shape': shape, 'channels': shape[target.axis],
                'parameters': target.original.size, 'source_dtype': str(target.original.dtype),
            })
    else:
        skipped.append({'reason': 'weight_only_requires_opset_13'})

    return {
        'file': path.name,
        'size_mib': round(path.stat().st_size / (1024**2), 3),
        'opset': [f"{i.domain or 'ai.onnx'}:{i.version}" for i in model.opset_import],
        'nodes': len(model.graph.node),
        'initializers': len(model.graph.initializer),
        'inputs': [v.name for v in model.graph.input],
        'outputs': [v.name for v in model.graph.output],
        'operators': dict(counts.most_common()),
        'candidates': candidates,
        'skipped': dict(Counter(item['reason'] for item in skipped)),
        'candidate_parameters': sum(c['parameters'] for c in candidates),
    }
