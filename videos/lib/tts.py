"""视频配音送去合成前的文字改写，所有视频共用（规范见 docs/tts.md）。
只改送给 TTS 的写法，配音稿与字幕不变。每条是 (正则, 替换)；新发现的误读先加到这里，再重录受影响的幕。"""
import re

COMMON = [
    # 运算与分数
    (r"1/(\d+)", r"\1 分之一"), (r" × ", " 乘 "),
    # 单位：字母单位会被逐字母读，统一写成中文
    (r"(?<=\d) W\b", " 瓦"), (r" pJ", " 皮焦"), (r" ns", " 纳秒"), (r" ms", " 毫秒"),
    # 数值格式的缩写：字母与数字分开读
    (r"\b(FP|TF|BF)(\d+)", r"\1 \2"), (r"INT8", "int 8"), (r"TFLOPS", "T FLOPS"),
]


def say(text, extra=()):
    """extra 是这部视频自己的术语读法，排在通用规则之后。"""
    for pattern, repl in [*COMMON, *extra]:
        text = re.sub(pattern, repl, text)
    # 去掉中文与英文、数字之间的空格（会被读成停顿）；英文词之间的空格保留
    return re.sub(r"(?<=[^\x00-\x7f]) (?=[\x21-\x7e])|(?<=[\x21-\x7e]) (?=[^\x00-\x7f])", "", text)
