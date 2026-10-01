# 生成 engine-figure-styles.html：vLLM 引擎文章的三张核心图（时间线 / 进程路径 / CUDA Graph 分派）× 10 种风格
import json, random, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
steps = json.load(open(ROOT / 'src/data/vllm-cpu-overhead-measured.json'))['steps']

SANS = "-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif"
MONO = "'SF Mono',Menlo,'PingFang SC',monospace"
SERIF = "'Songti SC','Noto Serif SC',Georgia,serif"

# hue 语义：gpu 计算 / idle 空转与 attention / core 关键路径 / off 不在关键路径 / io 线程 / disp dispatcher / pre prefill token
BASE = dict(mode='solid', bg='#fff', text='#1f2328', muted='#6b7280', rule='#e5e7eb', rx=8, font=SANS, hfont=None,
            fw=600, lw=2, on={}, cont=None, grid=False, shadow=False)

THEMES = [
    ('鲜明实心', '饱和实心色块 + 白字，淡色进程底带；最接近现有规范，只把颜色和字号拉开', dict(
        c=dict(gpu='#2f6bff', idle='#ff4d5e', core='#7a3ff2', off='#0fa38f', io='#dfe3ea', disp='#1a9c5b', pre='#f59f00'),
        on=dict(io='#374151', pre='#3b2400'), cont='tint')),
    ('淡底描边', '浅色填充 + 同色细描边 + 同色字，文档站常见的克制风格', dict(
        mode='tint', c=dict(gpu='#2563eb', idle='#dc2626', core='#7c3aed', off='#0d9488', io='#64748b', disp='#16a34a', pre='#d97706'),
        cont='dash', rx=6, fw=600, lw=1.6)),
    ('深色 IDE', '深底，亮色块，关键路径发亮的橙；夜间模式天然好看', dict(
        bg='#0d1117', text='#e6edf3', muted='#8b949e', rule='#30363d',
        c=dict(gpu='#58a6ff', idle='#ff7b72', core='#d2a8ff', off='#3fb950', io='#30363d', disp='#56d364', pre='#e3b341'),
        on=dict(gpu='#06142b', idle='#2b0906', core='#1f0b3a', off='#04210c', io='#c9d1d9', disp='#04210c', pre='#2b1d00'),
        cont='dark', accent='#ffa657')),
    ('新粗野', '粗黑描边 + 硬投影 + 高饱和平涂，直角，很有冲击力', dict(
        mode='brutal', c=dict(gpu='#4d7cff', idle='#ff5c5c', core='#b388ff', off='#2ee6a8', io='#f2f2f2', disp='#ffd23f', pre='#ffd23f'),
        on=dict(gpu='#000', idle='#000', core='#000', off='#000', io='#000', disp='#000', pre='#000'),
        rx=2, fw=800, lw=2.5, cont='brutal', accent='#000', bg='#fffdf5')),
    ('蓝图', '藏青底 + 网格 + 白色线框 + 等宽字，工程图纸感', dict(
        mode='outline', bg='#0b2a4a', text='#e8f1ff', muted='#8fb3d9', rule='#28507a', font=MONO, grid=True,
        c=dict(gpu='#7fd1ff', idle='#ff8a80', core='#ffffff', off='#9be7c4', io='#8fb3d9', disp='#ffe28a', pre='#ffe28a'),
        cont='bp', rx=2, lw=1.6, accent='#ffe28a')),
    ('编辑印刷', '米白纸底、宋体标题、墨黑 + 一种朱红强调，像杂志插图', dict(
        bg='#faf7f0', text='#1b1b1b', muted='#6f6a60', rule='#d8d0c0', hfont=SERIF,
        c=dict(gpu='#1b1b1b', idle='#d7462b', core='#d7462b', off='#8a8478', io='#e7e0d2', disp='#1b1b1b', pre='#b9ab8f'),
        on=dict(io='#1b1b1b', pre='#1b1b1b'), cont='rule', rx=0, fw=600, accent='#d7462b')),
    ('灰阶 + 单一强调', '全部中性灰，只有关键路径 / 本例用亮橙，一眼看到重点', dict(
        c=dict(gpu='#3d4652', idle='#ff5a1f', core='#ff5a1f', off='#aab2bd', io='#e3e7ec', disp='#3d4652', pre='#aab2bd'),
        on=dict(io='#3d4652', off='#1f2328', pre='#1f2328'), cont='plain', rx=10, accent='#ff5a1f')),
    ('马卡龙柔和', '低饱和粉彩、大圆角胶囊、深色字，亲和不刺眼', dict(
        c=dict(gpu='#a7c7ff', idle='#ffb4b4', core='#cdb8ff', off='#a8e6cf', io='#eceff3', disp='#bfe8b0', pre='#ffe0a3'),
        on={k: '#2b2f38' for k in ('gpu', 'idle', 'core', 'off', 'io', 'disp', 'pre')},
        cont='pastel', rx=18, accent='#8a63f0', fw=600)),
    ('手绘草图', 'Excalidraw 式抖动线条 + 半透明填充，适合讲机制的“白板”感', dict(
        mode='sketch', c=dict(gpu='#1971c2', idle='#e03131', core='#7048e8', off='#0c8599', io='#868e96', disp='#2f9e44', pre='#f08c00'),
        cont='sketch', rx=4, lw=1.8, accent='#7048e8')),
    ('浮起卡片', '白卡片 + 柔和阴影 + 彩色顶边，Apple / Vercel 式产品图', dict(
        mode='card', bg='#f3f4f6', c=dict(gpu='#0a84ff', idle='#ff375f', core='#bf5af2', off='#30b0c7', io='#8e8e93', disp='#34c759', pre='#ff9f0a'),
        cont='card', rx=12, shadow=True, fw=600)),
]


def T(t):
    t = {**BASE, **t}
    t.setdefault('accent', t['c']['core'])
    t['hfont'] = t['hfont'] or t['font']
    return t


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')


def txt(t, x, y, s, fill=None, fs=13, a='middle', fw=400, head=False):
    return (f'<text x="{x}" y="{y}" fill="{fill or t["text"]}" font-size="{fs}" font-weight="{fw}" text-anchor="{a}" '
            f'dominant-baseline="central" font-family="{t["hfont"] if head else t["font"]}">{esc(s)}</text>')


def rough(x, y, w, h, seed, amp=1.6):
    rnd = random.Random(seed)
    j = lambda: rnd.uniform(-amp, amp)
    pts = [(x + j(), y + j()), (x + w + j(), y + j()), (x + w + j(), y + h + j()), (x + j(), y + h + j())]
    d = f'M{pts[0][0]:.1f} {pts[0][1]:.1f}'
    for i in range(1, 5):
        p, q = pts[i - 1], pts[i % 4]
        mx, my = (p[0] + q[0]) / 2 + j(), (p[1] + q[1]) / 2 + j()
        d += f' Q{mx:.1f} {my:.1f} {q[0]:.1f} {q[1]:.1f}'
    return d


def box(t, x, y, w, h, hue, label='', fs=13, sub=None, data=False):
    c = t['c'][hue] if hue != 'empty' else None
    rx, m = min(t['rx'], h / 2), t['mode']
    if data and m == 'card':  # 数据条不做成卡片
        m, rx = 'solid', 4
    tc, o = t['on'].get(hue, '#fff'), ''
    if hue == 'empty':
        o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(rx, 4)}" fill="none" stroke="{t["muted"]}" stroke-width="1.2" stroke-dasharray="3 2"/>'
        return o
    if m == 'solid':
        o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c}"/>'
    elif m == 'tint':
        o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c}" fill-opacity=".12" stroke="{c}" stroke-width="{t["lw"]}"/>'
        tc = c
    elif m == 'outline':
        o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c}" fill-opacity=".10" stroke="{c}" stroke-width="{t["lw"]}"/>'
        tc = c
    elif m == 'brutal':
        o = (f'<rect x="{x + 4}" y="{y + 4}" width="{w}" height="{h}" rx="{rx}" fill="#000"/>'
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c}" stroke="#000" stroke-width="{t["lw"]}"/>')
    elif m == 'sketch':
        s = hash((x, y, w, h)) & 0xffff
        o = (f'<path d="{rough(x, y, w, h, s, 0)}" fill="{c}" fill-opacity=".18"/>'
             f'<path d="{rough(x, y, w, h, s)}" fill="none" stroke="{c}" stroke-width="{t["lw"]}" stroke-linecap="round"/>'
             '')
        tc = c
    elif m == 'card':
        o = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="#fff" filter="url(#sh)"/>'
             f'<rect x="{x}" y="{y}" width="{min(5, w)}" height="{h}" rx="2" fill="{c}"/>' if w > 30 else
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(rx, 4)}" fill="{c}"/>')
        tc = t['text'] if w > 30 else '#fff'
    if label:
        cy = y + h / 2 - (8 if sub else 0)
        o += txt(t, x + w / 2, cy, label, tc, fs, fw=t['fw'])
        if sub:
            o += txt(t, x + w / 2, cy + 18, sub, tc, 11, fw=400)
    return o


def container(t, x, y, w, h, title, hue):
    c, k = t['c'][hue], t['cont']
    tt = txt(t, x + 14, y + 20, title, c if k not in ('rule', 'plain', 'card', 'brutal') else t['text'], 13, 'start', 700, head=True)
    if k == 'tint':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c}" fill-opacity=".07"/>'
    elif k == 'dash':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="{c}" stroke-opacity=".5" stroke-dasharray="5 4"/>'
    elif k == 'dark':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#161b22" stroke="#30363d"/>'
    elif k == 'brutal':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="#fff" stroke="#000" stroke-width="2.5"/>'
    elif k == 'bp':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{t["muted"]}" stroke-dasharray="6 4"/>'
    elif k == 'rule':
        r = f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke="{t["text"]}" stroke-width="2"/>'
    elif k == 'plain':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#f4f6f8"/>'
    elif k == 'pastel':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{c}" fill-opacity=".22"/>'
        tt = txt(t, x + 16, y + 20, title, '#2b2f38', 13, 'start', 700)
    elif k == 'sketch':
        r = f'<path d="{rough(x, y, w, h, x + y, 2.4)}" fill="none" stroke="{c}" stroke-width="1.4" stroke-dasharray="7 5" stroke-linecap="round"/>'
    elif k == 'card':
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="#fafafb" stroke="#e5e7eb"/>'
    return r + tt


HEAD = 8  # 箭头长度：路径终点写在目标边上，线条在箭头底部截止，由箭头尖端落到边上


def shorten(d, n=HEAD):
    # 只含 M / H / V 的正交路径：把最后一段缩短 n，线不伸进箭头
    toks = d.replace('M', ' M ').replace('H', ' H ').replace('V', ' V ').split()
    x = y = 0
    i = 0
    while i < len(toks):
        c = toks[i]
        if c == 'M': x, y = float(toks[i + 1]), float(toks[i + 2]); i += 3; continue
        v = float(toks[i + 1])
        if i + 2 >= len(toks):
            prev = x if c == 'H' else y
            toks[i + 1] = f'{v - n if v > prev else v + n:g}'
        if c == 'H': x = v
        else: y = v
        i += 2
    return ' '.join(toks)


def link(t, d, focus=False, head=True):
    c = t['accent'] if focus else t['muted']
    w = t['lw'] + (1.2 if focus else 0)
    extra = ' stroke-linecap="butt" stroke-linejoin="round"'
    if not head:
        return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{extra}/>'
    return f'<path d="{shorten(d)}" fill="none" stroke="{c}" stroke-width="{w}"{extra} marker-end="url(#{"af" if focus else "am"})"/>'


def badge(t, x, y, n):
    c = t['accent']
    st = ' stroke="#000" stroke-width="2"' if t['mode'] == 'brutal' else ''
    tc = '#0b2a4a' if t['cont'] == 'bp' else ('#fff' if t['mode'] != 'brutal' else '#fff')
    return f'<circle cx="{x}" cy="{y}" r="10"{st} fill="{c}"/>' + txt(t, x, y + .5, n, tc, 11, fw=700)


def legend(t, x, y, items):
    o = ''
    for hue, label in items:
        o += box(t, x, y - 6, 12, 12, hue) + txt(t, x + 18, y, label, t['muted'], 11.5, 'start')
        x += 30 + len(label) * 11.5
    return o


def svg(t, i, h, body):
    grid = ''
    if t['grid']:
        grid = (f'<pattern id="g{i}" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0 H0 V20" fill="none" stroke="#1d4570" stroke-width="1"/></pattern>'
                f'')
    defs = (f'<defs>{grid}'
            f'<marker id="af" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1.5 L9 5 L1 8.5 Z" fill="{t["accent"]}"/></marker>'
            f'<marker id="am" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1.5 L9 5 L1 8.5 Z" fill="{t["muted"]}"/></marker>'
            f'<filter id="sh" x="-10%" y="-20%" width="120%" height="150%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity=".12"/></filter></defs>')
    bg = f'<rect x="-20" y="-20" width="800" height="{h + 40}" fill="{f"url(#g{i})" if t["grid"] else t["bg"]}"/>'
    out = (f'<svg viewBox="-20 -20 800 {h + 40}" style="display:block;width:100%;background:{t["bg"]};border-radius:10px">'
            f'{defs}{bg}{body}</svg>')
    for k in ('af', 'am', 'sh'):
        out = out.replace(f'id="{k}"', f'id="{k}{i}"').replace(f'url(#{k})', f'url(#{k}{i})')
    return out


# 图 1：step timeline
def fig_timeline(t, i):
    PX, X0 = 17, 150
    b = txt(t, 0, 0, '时间 →（等比例）', t['muted'], 11.5, 'start')
    for r, (label, key) in enumerate([('关 async scheduling', 'no_async/c8'), ('默认（async）', 'default/c8')]):
        s = steps[key]
        busy, idle = s['gpu_busy'] / 1000 * PX, (s['period'] - s['gpu_busy']) / 1000 * PX
        y = 24 + r * 64
        b += txt(t, 0, y + 18, label, fs=13, a='start', fw=600, head=True)
        for n in range(3):
            x = X0 + n * (busy + idle)
            b += box(t, x, y, busy - 2, 36, 'gpu', data=True, label= f'{s["gpu_busy"] / 1000:.1f} ms' if n == 0 else '')
            b += box(t, x + busy, y, idle - 2, 36, 'idle', data=True, fs=12, label= f'{(s["period"] - s["gpu_busy"]) / 1000:.1f}' if n == 0 and idle > 30 else '')
        if idle <= 30:
            b += txt(t, X0 + busy + idle / 2, y + 48, f'{(s["period"] - s["gpu_busy"]) / 1000:.1f}', t['muted'], 11)
        b += txt(t, X0 + 3 * (busy + idle) + 8, y + 18, f'一步 {s["period"] / 1000:.1f} ms', t['text'], 12.5, 'start', 600)
    b += legend(t, 150, 168, [('gpu', 'GPU 计算'), ('idle', 'GPU 空转')])
    return svg(t, i * 10 + 1, 180, b)


# 图 2：一个请求穿过的进程与线程
def fig_process(t, i):
    b = container(t, 0, 0, 222, 312, '进程 1 · API server', 'off')
    b += container(t, 240, 0, 340, 312, '进程 2 · EngineCore', 'core')
    b += container(t, 598, 0, 162, 312, 'GPU', 'gpu')
    b += box(t, 18, 52, 186, 54, 'off', 'HTTP · tokenize')
    b += box(t, 18, 223, 186, 60, 'off', '增量 detokenize', sub='流式返回')
    b += box(t, 258, 58, 150, 42, 'io', '输入线程')
    b += box(t, 258, 124, 304, 76, 'core', '主线程 busy loop', fs=15, sub='schedule → 准备输入 → 提交')
    b += box(t, 412, 226, 150, 54, 'io', '输出线程')
    b += box(t, 616, 124, 126, 76, 'gpu', 'forward', sub='+ sample')
    b += link(t, 'M204 79 H258', True) + link(t, 'M333 100 V124', True) + link(t, 'M562 162 H616', True)
    b += link(t, 'M679 200 V253 H562', True) + link(t, 'M412 253 H204', True)
    b += txt(t, 231, 92, 'ZMQ', t['muted'], 10.5) + txt(t, 300, 268, 'ZMQ', t['muted'], 10.5)
    for n, (x, y) in enumerate([(231, 64), (349, 112), (589, 146), (695, 226), (372, 238), (111, 298)], 1):
        b += badge(t, x, y, str(n))
    b += txt(t, 18, 146, 'tokenize、detokenize 与', t['muted'], 11.5, 'start') + txt(t, 18, 164, '网络 IO 都不占主线程', t['muted'], 11.5, 'start')
    b += legend(t, 60, 334, [('off', '不在关键路径的 CPU'), ('core', '关键路径 CPU'), ('io', 'IO 线程'), ('gpu', 'GPU')])
    return svg(t, i * 10 + 2, 346, b)


# 图 3：CUDA Graph 分派
def fig_dispatch(t, i):
    b = txt(t, 0, 0, '本步 batch：3 decode + 1 prefill chunk = 13 token → 补齐到 16', fs=13, a='start', fw=600, head=True)
    for k in range(16):
        hue = 'gpu' if k < 3 else 'pre' if k < 13 else 'empty'
        b += box(t, k * 30, 20, 26, 26, hue, 'd' if k < 3 else 'p' if k < 13 else '', fs=11)
    b += txt(t, 492, 33, '← 3 个 padding', t['muted'], 11.5, 'start')
    b += box(t, 225, 78, 330, 44, 'disp', 'dispatcher：token 数 · 是否 uniform decode')
    b += link(t, 'M253 46 V78') + link(t, 'M100 146 H660', head=False)
    b += link(t, 'M100 146 V166') + link(t, 'M660 146 V166') + link(t, 'M390 122 V166', True)
    for x, name, cond, hue in [(0, 'FULL 整图', '纯 uniform decode 且有对应 size', 'gpu'),
                               (290, 'PIECEWISE 分段图', 'prefill / 混合 batch（本例）', 'core'),
                               (560, 'eager', '没有可用的 capture size', 'io')]:
        b += container(t, x, 166, 200, 150, name, hue)
        b += txt(t, x + 14, 208, cond, t['muted'], 11, 'start')
    b += box(t, 14, 232, 172, 30, 'gpu', '整条 forward 含 attention', fs=11.5, data=True)
    b += txt(t, 14, 290, 'replay 一次 = 一次 launch', t['muted'], 11, 'start')
    for x in (304, 372, 440):
        b += box(t, x, 232, 44, 30, 'gpu', data=True)
    for x in (350, 418):
        b += box(t, x, 232, 20, 30, 'idle', data=True)
    b += txt(t, 304, 282, 'attention 形状随请求变化，', t['muted'], 11, 'start') + txt(t, 304, 298, '留在图外逐 kernel launch', t['muted'], 11, 'start')
    for k in range(10):
        b += box(t, 574 + k * 17.5, 232, 13, 30, 'io', data=True)
    b += txt(t, 574, 290, '每个 kernel 单独 launch', t['muted'], 11, 'start')
    b += legend(t, 150, 342, [('gpu', '进图的部分'), ('idle', 'attention（逐 kernel）'), ('io', 'eager launch')])
    return svg(t, i * 10 + 3, 354, b)


cards = ''
for i, (name, desc, cfg) in enumerate(THEMES):
    t = T(cfg)
    cards += (f'<section><h2><span>{i + 1:02d}</span>{name}</h2><p>{desc}</p>'
              f'<div class="row"><div>{fig_process(t, i)}</div><div>{fig_dispatch(t, i)}</div></div>'
              f'<div class="tl">{fig_timeline(t, i)}</div></section>')

html = f'''<!doctype html><meta charset="utf-8"><title>Engine Figure Styles</title>
<style>
body{{margin:0;padding:32px 40px 80px;background:#eceef1;font-family:{SANS};color:#1f2328}}
h1{{font-size:22px;margin:0 0 6px}} .lead{{color:#6b7280;margin:0 0 28px;font-size:14px}}
section{{background:#fff;border-radius:14px;padding:20px 22px 22px;margin-bottom:28px}}
h2{{font-size:18px;margin:0 0 4px;display:flex;align-items:center;gap:10px}}
h2 span{{font:600 13px/1 {MONO};background:#1f2328;color:#fff;border-radius:6px;padding:5px 7px}}
section>p{{color:#6b7280;font-size:13px;margin:0 0 14px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .tl{{margin-top:18px;max-width:calc(50% - 9px)}}
</style>
<h1>vLLM 引擎文章 · 图表风格 10 选 1</h1>
<p class="lead">同一组内容（进程路径 / CUDA Graph 分派 / step 时间线，时间线用实测数据）换 10 种视觉语言。选定后按该风格重做 figure-kit 与全文五张图，并补夜间配色。</p>
{cards}'''
(pathlib.Path(__file__).parent / 'engine-figure-styles.html').write_text(html)
print('ok')
