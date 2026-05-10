#src/ml/__init__.py

"""
ML package for 2D CNN-based RF signal classification.

현재 역할:
- STFT spectrogram을 CNN 입력 형태로 맞춤
- dummy 또는 keras classifier를 생성
- CNN 결과를 class_name / confidence 형태로 반환

현재 프로젝트 기준:
- CNN 입력 shape: (512, 125, 1)
- class: Background / WiFi / Bluetooth / Drone-like
"""
from src.ml.rf3_labels import (
    LABEL_TO_ID,
    ID_TO_LABEL,
    label_to_id,
    id_to_label,
    num_rf3_classes,
)

from src.ml.rf3_dataset import (
    RFSpectrogramDataset,
    read_manifest_csv,
    compute_spectrogram_mean_std,
)

from src.ml.rf3_model import (
    RF3SmallCNN,
)

from src.ml.rf3_metrics import (
    build_confusion_matrix,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    make_classification_report_text,
    save_text,
)
from src.ml.inference import (
    CNNResult,
    DummyCNNClassifier,
    KerasCNNClassifier,
    build_cnn_classifier,
)

from src.ml.transforms import (
    ensure_cnn_input_shape,
    add_batch_dimension,
)

from src.ml.rf3_inference import (
    RF3Classifier,
    RF3Result,
)

__all__ = [
    "CNNResult",
    "DummyCNNClassifier",
    "KerasCNNClassifier",
    "build_cnn_classifier",
    "ensure_cnn_input_shape",
    "add_batch_dimension",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "label_to_id",
    "id_to_label",
    "num_rf3_classes",
    "RFSpectrogramDataset",
    "read_manifest_csv",
    "compute_spectrogram_mean_std",
    "RF3SmallCNN",
    "build_confusion_matrix",
    "save_confusion_matrix_csv",
    "save_confusion_matrix_png",
    "make_classification_report_text",
    "save_text",
    "RF3Classifier",
    "RF3Result",
]
