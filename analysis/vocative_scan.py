"""Scan every logged decode for «يا + given name» (pre-registered side-talk rule, Nov 21)."""
import re, sys, glob
sys.path.insert(0, ".")
from turn_rules import normalize, tokens
NAMES = set(normalize(n) for n in """محمد أحمد احمد عبدالله عبدالرحمن عبدالعزيز عبدالمجيد عبدالكريم خالد فهد سعود سلطان فيصل تركي ناصر سعد
عمر علي يوسف إبراهيم ابراهيم ماجد بندر نايف مشعل مساعد حمد راشد سالم صالح منصور ياسر وليد طلال زياد مازن ريان نواف
بدر عادل حسن حسين مصطفى حمود متعب عبدالرحيم سامي هاني نورة نوره سارة ساره فاطمة فاطمه مريم ريم هند لولوة منيرة
هيا العنود الجوهرة دانة رهف لمى شهد نوف أسماء اسماء عائشة خلود أمل امل منى ابتسام لطيفة غادة""".split())
VOC = re.compile(r"(?:^| )يا+ (\S+)")
def vocative_name(text):
    for m in VOC.finditer(" ".join(tokens(text))):
        if m.group(1) in NAMES:
            return m.group(1)
    return None
if __name__ == "__main__":
    pat = re.compile(r"'(.*)' -> (STOP|keep) \((.*)\)")
    n = 0
    for f in ["asr_log.txt", "demo_log.txt", "eleven_log.txt"]:
        try: lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        except OSError: continue
        for l in lines:
            m = pat.search(l)
            if not m: continue
            n += 1
            nm = vocative_name(m.group(1))
            if nm: print(f, nm, "|", m.group(1)[:80], "|", m.group(2), m.group(3))
    print("decodes scanned:", n)
