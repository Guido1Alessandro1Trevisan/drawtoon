# ----------------------------------------------------------------------------
# Code based on pixart-alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/diffusion_utils.py
# ----------------------------------------------------------------------------

import html
import os
import re
import urllib.parse as ul
from typing import Any, List, Union

import ftfy
import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5EmbedderGoogle:
    """
    Example:
        import torch
        from layousyn.model.t5 import T5EmbedderGoogle
        t5 = T5EmbedderGoogle(device="cuda", local_cache=False, model_max_length=120)
        t5 = T5EmbedderGoogle(device="cuda", local_cache=False, cache_dir=args.t5_path, torch_dtype=torch.float)
        output = t5.get_text_embeddings(["Hello World!"])

        Note: Set local_cache=False if you want to download the model from huggingface
    """

    def __init__(
        self,
        device,
        dir_or_name,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=torch.bfloat16,
        model_max_length=120,
        is_torch_compile=True,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        self.dir_or_name = f"google/{dir_or_name}"
        self.model_max_length = model_max_length

        t5_model_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": self.torch_dtype}
        t5_model_kwargs["device_map"] = {"shared": self.device, "encoder": self.device}
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir_or_name)
        self.model = T5EncoderModel.from_pretrained(
            self.dir_or_name, **t5_model_kwargs
        ).eval()
        self.model = torch.compile(self.model) if is_torch_compile else self.model

    @torch.no_grad()
    def get_tokens(self, texts) -> Union[List[str], Any, Any]:
        processed_texts = [self.text_preprocessing(text) for text in texts]
        text_tokens_and_mask = self.tokenizer(
            processed_texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        tokens = text_tokens_and_mask["input_ids"]
        attention_mask = text_tokens_and_mask["attention_mask"]
        padding_mask = attention_mask == 0

        return processed_texts, tokens, padding_mask

    @torch.no_grad()
    def get_text_embeddings_from_tokens(self, input_ids, padding_mask):
        attention_mask = padding_mask == 0
        text_encoder_embeds = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )["last_hidden_state"].detach()
        return text_encoder_embeds

    def get_text_embeddings(self, texts):
        # tokenization
        processed_texts, input_ids, padding_mask = self.get_tokens(texts)
        input_ids = input_ids.to(self.device)  # type: ignore
        padding_mask = padding_mask.to(self.device)  # type: ignore

        # inference
        text_encoder_embs = self.get_text_embeddings_from_tokens(
            input_ids, padding_mask
        )
        return processed_texts, text_encoder_embs, padding_mask

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @staticmethod
    def clean_caption(caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls

        # # html:
        # caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + "\)"
            + "\("
            + "\]"
            + "\["
            + "\}"
            + "\{"
            + "\|"
            + "\\"
            + "\/"
            + "\*"
            + r"]{1,}"
        )  # noqa
        caption = re.sub(
            bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = T5EmbedderGoogle.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    @staticmethod
    def text_preprocessing(text):
        # The exact text cleaning as was in the training stage:
        text = T5EmbedderGoogle.clean_caption(text)
        text = T5EmbedderGoogle.clean_caption(text)
        return text
