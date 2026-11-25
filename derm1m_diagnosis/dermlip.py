"""
Signal B / C：DermLIP（Derm1M 的 CLIP 风格模型）视觉零样本 + 文本检索。

懒加载 + 优雅降级：如果 open_clip / 权重 / 网络任一不可用，
available 置为 False，所有方法返回 None，管线自动退化为纯符号规则模式。

用法：
    client = DermLIPClient()
    probs = client.zero_shot_disease(image_pil)          # -> {disease: prob} 或 None
    texts  = client.retrieve_text(query_text, corpus)    # -> [(text, score)] 或 None

注意：DermLIP 是 CLIP 模型，预处理归一化是 OpenAI CLIP 标准，必须用
create_model_and_transforms 返回的 preprocess_val，不能用 ImageNet 归一化。
"""

from pathlib import Path

# Derm1M 仓库根（含 src/open_clip）
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DERM1M_SRC = _REPO_ROOT / "Derm1M" / "src"

# 默认用 DermLIP-PanDerm（Derm1M 论文里零样本更强）；可改 DermLIP_ViT-B-16
DEFAULT_MODEL = "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256"
# DermLIP_PanDerm-base-w-PubMed-256 或 DermLIP_ViT-B-16

# 皮肤诊断用的 prompt 模板（参考 Derm1M/concept_annotation/model.py）
SKIN_TEMPLATES = [
    "This is a skin image of {}",
    "A skin image of {}",
    "An image of {}, a skin condition.",
    "{}, a skin disorder, is shown in this image.",
    "The skin lesion depicted is {}.",
]


class DermLIPClient:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or "cpu"
        self.available = False
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_error = None

    def load(self) -> bool:
        """加载 DermLIP。成功返回 True；失败记下错误并保持 available=False。"""
        if self._model is not None:
            return True
        try:
            import sys
            import torch
            if str(_DERM1M_SRC) not in sys.path:
                sys.path.insert(0, str(_DERM1M_SRC))

            from open_clip import create_model_and_transforms, get_tokenizer

            model, _, preprocess = create_model_and_transforms(
                self.model_name, device=self.device,
            )
            tokenizer = get_tokenizer(self.model_name)
            model.eval()

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self.available = True
            return True
        except Exception as e:  # noqa: BLE001 —— 网络/依赖/权重任一失败都降级
            self._load_error = f"{type(e).__name__}: {e}"
            self.available = False
            return False

    # ------------------------------------------------------------------ 编码
    def _encode_image(self, image_pil):
        import torch
        with torch.no_grad():
            img = self._preprocess(image_pil).unsqueeze(0).to(self.device)
            feats = self._model.encode_image(img, normalize=True)
        return feats  # [1, D]

    def _encode_text(self, texts: list[str]):
        import torch
        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self.device)
            feats = self._model.encode_text(tokens, normalize=True)
        return feats  # [N, D]

    # ----------------------------------------------------- Signal B：零样本
    def zero_shot_disease(self, image_pil, classnames=None) -> dict[str, float] | None:
        """整图零样本三分类。返回 {disease: prob}，不可用返回 None。"""
        if not self.available and not self.load():
            return None
        import torch
        classnames = classnames or ["eczema", "bullous pemphigoid", "urticaria"]

        texts = [t.format(c) for c in classnames for t in SKIN_TEMPLATES]
        with torch.no_grad():
            text_feats = self._encode_text(texts)          # [C*T, D]
            img_feats = self._encode_image(image_pil)      # [1, D]
            logits = img_feats @ text_feats.T              # [1, C*T]
            logits = logits.reshape(1, len(classnames), -1).mean(dim=-1)  # [1, C]
            probs = logits.softmax(dim=-1)[0].cpu().numpy()

        return {c: float(p) for c, p in zip(classnames, probs)}

    def zero_shot_lesion_crop(self, image_pil, classnames=None) -> dict[str, float] | None:
        """对单个病灶裁剪图做零样本（概念级），可用于 Stage1 的校准。"""
        if not self.available and not self.load():
            return None
        import torch
        classnames = classnames or [
            "macule", "patch", "papule", "plaque", "vesicle",
            "bulla", "scale", "crust", "erosion",
        ]
        texts = [t.format(c) for c in classnames for t in SKIN_TEMPLATES]
        with torch.no_grad():
            text_feats = self._encode_text(texts)
            img_feats = self._encode_image(image_pil)
            logits = img_feats @ text_feats.T
            logits = logits.reshape(1, len(classnames), -1).mean(dim=-1)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(classnames, probs)}

    # ------------------------------------------- Signal C：文本诊断检索
    def retrieve_text(self, query_text: str, corpus: list[str], top_k: int = 3):
        """文本检索：把皮损摘要句编码后，在诊断描述语料库中找最相似文本。

        返回 [(text, score)]，不可用返回 None。
        注意：DermLIP 文本编码器是为图文对齐训练的，text→text 检索是近似用法，
        需在真实数据上验证效果，不能当作严谨的推理。
        """
        if not self.available and not self.load():
            return None
        import torch
        with torch.no_grad():
            q = self._encode_text([query_text])       # [1, D]
            d = self._encode_text(corpus)             # [N, D]
            sim = (q @ d.T)[0]                        # [N]
            top = sim.topk(min(top_k, len(corpus)))
        return [(corpus[i], float(s)) for i, s in zip(top.indices, top.values)]
