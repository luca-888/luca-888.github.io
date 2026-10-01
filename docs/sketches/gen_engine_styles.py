# 生成 engine-process-styles.html：同一张“进程与线程”图的 8 种风格样稿
F = 'font-family="-apple-system,BlinkMacSystemFont,\'PingFang SC\',sans-serif"'
def box(x,y,w,h,fill,label,tc='#fff',rx=8,fs=14,stroke=None,fw=600):
    st=f' stroke="{stroke}" stroke-width="2"' if stroke else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{st}/><text x="{x+w/2}" y="{y+h/2+0.5}" fill="{tc}" font-size="{fs}" font-weight="{fw}" text-anchor="middle" dominant-baseline="central" {F}>{label}</text>'
def txt(x,y,t,c='#555',fs=12,a='middle',fw=400):
    return f'<text x="{x}" y="{y}" fill="{c}" font-size="{fs}" font-weight="{fw}" text-anchor="{a}" dominant-baseline="central" {F}>{t}</text>'
def arr(d,c,w=2,dash=None,id='a'):
    da=f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{da} marker-end="url(#{id})"/>'
def defs(id,c):
    return f'<marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1.5 L9 5 L1 8.5 Z" fill="{c}"/></marker>'
def svg(i,bg,body,dc,h=380):
    return f'<svg viewBox="0 0 760 {h}" style="background:{bg};border-radius:10px;width:100%"><defs>{defs("a"+str(i),dc)}</defs>{body}</svg>'

S=[]
# 1 鲜明分行
c=dict(t='#0f9d8a',p='#7a3ff2',b='#2f6bff',g='#dde1e8')
b=''
for y,h,col in [(8,110,c['t']),(128,120,c['p']),(258,110,c['b'])]:
    b+=f'<rect x="0" y="{y}" width="760" height="{h}" rx="10" fill="{col}" fill-opacity=".11"/>'
b+=txt(16,28,'API server · 进程 1',c['t'],13,'start',700)+txt(16,148,'EngineCore · 进程 2',c['p'],13,'start',700)+txt(16,278,'GPU',c['b'],13,'start',700)
b+=box(130,48,130,52,c['t'],'tokenize')+box(590,48,150,52,c['t'],'detokenize · 流式返回')+box(130,176,130,52,c['g'],'输入线程','#333')+box(330,168,200,68,c['p'],'主线程 busy loop',fs=15)+box(590,176,150,52,c['g'],'输出线程','#333')+box(330,300,200,48,c['b'],'forward + sample')
a='a1';k='#444'
b+=arr('M195 100 V174',k,2,id=a)+arr('M260 202 H328',k,2,id=a)+arr('M400 236 V298',k,2,id=a)+arr('M460 298 V238',k,2,id=a)+arr('M530 202 H588',k,2,id=a)+arr('M665 174 V102',k,2,id=a)
b+=txt(203,138,'ZMQ','#666',11,'start')+txt(673,138,'ZMQ','#666',11,'start')+txt(392,268,'提交 batch','#666',11,'end')+txt(468,268,'采样结果','#666',11,'start')
S.append(('1 · 鲜明分行（U 形）','饱和实心色 + 淡色分区带，进程分行',svg(1,'#fff',b,k)))

# 2 深色底
bg='#0e1116'
b=''
b+=f'<line x1="0" y1="128" x2="760" y2="128" stroke="#2a313b"/><line x1="0" y1="260" x2="760" y2="260" stroke="#2a313b"/>'
b+=txt(0,70,'API server','#8b98a8',12,'start',600)+txt(0,200,'EngineCore','#8b98a8',12,'start',600)+txt(0,322,'GPU','#8b98a8',12,'start',600)
b+=box(120,44,150,52,'#1fd1b0','tokenize','#06201b')+box(560,44,170,52,'#1fd1b0','detokenize · 流式返回','#06201b')+box(120,174,150,52,'#2b323d','输入线程','#c8d1dc')+box(330,166,180,68,'#ff6b4a','主线程 busy loop','#1a0a05',fs=15)+box(560,174,170,52,'#2b323d','输出线程','#c8d1dc')+box(330,294,180,52,'#5aa0ff','forward + sample','#04162e')
k='#ff9d85';a='a2'
b+=arr('M195 96 V172',k,2.5,id=a)+arr('M270 200 H328',k,2.5,id=a)+arr('M400 234 V292',k,2.5,id=a)+arr('M460 292 V236',k,2.5,id=a)+arr('M510 200 H558',k,2.5,id=a)+arr('M645 174 V98',k,2.5,id=a)
b+=txt(205,134,'ZMQ','#8b98a8',11,'start')+txt(655,134,'ZMQ','#8b98a8',11,'start')
S.append(('2 · 深色底','深色背景，橙红只强调关键路径',svg(2,bg,b,k)))

# 3 嵌套容器 横向流水线
b=''
b+=f'<rect x="8" y="30" width="200" height="300" rx="16" fill="#e8f5f1"/><rect x="228" y="30" width="360" height="300" rx="16" fill="#efe9fb"/><rect x="608" y="30" width="144" height="300" rx="16" fill="#e5eefe"/>'
b+=txt(20,52,'API server','#0f7f70',13,'start',700)+txt(240,52,'EngineCore','#6a35d0',13,'start',700)+txt(620,52,'GPU','#2456d6',13,'start',700)
b+=box(24,90,168,56,'#0f9d8a','tokenize')+box(24,254,168,56,'#0f9d8a','detokenize · 返回')
b+=box(244,90,150,56,'#a78bdc','输入线程')+box(414,160,160,72,'#7a3ff2','主线程',fs=16)+box(244,254,150,56,'#a78bdc','输出线程')
b+=box(624,160,112,72,'#2f6bff','GPU forward')
k='#333';a='a3'
b+=arr('M192 118 H244',k,2,id=a)+arr('M394 118 H430 V158',k,2,id=a)+arr('M574 184 H622',k,2,id=a)+arr('M622 212 H576',k,2,id=a)+arr('M470 232 V282 H396',k,2,id=a)+arr('M244 282 H194',k,2,id=a)
b+=txt(218,104,'ZMQ','#555',10)+txt(219,268,'ZMQ','#555',10)
S.append(('3 · 嵌套容器','进程是大圆角容器，线程是里面的色块，路径蛇形',svg(3,'#fff',b,k)))

# 4 时序图
b=''
cols=[('API server','#0f9d8a',90),('输入线程','#8a94a3',240),('主线程','#7a3ff2',400),('输出线程','#8a94a3',560),('GPU','#2f6bff',690)]
for n,cl,x in cols:
    b+=box(x-55,12,110,36,cl,n,fs=13)+f'<line x1="{x}" y1="48" x2="{x}" y2="368" stroke="{cl}" stroke-opacity=".45" stroke-dasharray="4 4"/>'
k='#222';a='a4'
def msg(y,x1,x2,t,n):
    d=f'M{x1} {y} H{x2 + (-1 if x2>x1 else 1)*2}'
    return arr(d,k,2,id=a)+txt((x1+x2)/2,y-11,f'{n} {t}','#333',12,'middle',600)
b+=msg(84,90,240,'tokenize 后经 ZMQ',1)+msg(126,240,400,'请求入队',2)+msg(168,400,690,'提交 batch',3)
b+=f'<rect x="680" y="176" width="20" height="44" rx="3" fill="#2f6bff"/>'
b+=msg(240,690,400,'采样结果',4)+msg(282,400,560,'结果入队',5)+msg(324,560,90,'ZMQ · detokenize · 流式返回',6)
S.append(('4 · 时序图','泳道 + 消息，时间自上而下，编号即步骤',svg(4,'#fff',b,k)))

# 5 暖色纸感
bg='#faf5ea'
b=''
b+=f'<rect x="12" y="12" width="736" height="96" rx="22" fill="#f1e2c4"/><rect x="12" y="126" width="736" height="122" rx="22" fill="#eed3c4"/><rect x="12" y="266" width="736" height="102" rx="22" fill="#d9e3cf"/>'
b+=txt(32,32,'API server','#8a6a2b',13,'start',700)+txt(32,146,'EngineCore','#a4492b',13,'start',700)+txt(32,286,'GPU','#4d6a3a',13,'start',700)
b+=box(150,44,130,50,'#d9a441','tokenize','#3b2a08',25)+box(590,44,150,50,'#d9a441','detokenize · 流式返回','#3b2a08',25)
b+=box(150,174,130,50,'#c9c0b0','输入线程','#3b3020',25)+box(330,164,190,70,'#c8553d','主线程 busy loop','#fff',35,15)+box(590,174,150,50,'#c9c0b0','输出线程','#3b3020',25)
b+=box(330,298,190,50,'#6f8f4e','forward + sample','#fff',25)
k='#5b4636';a='a5'
b+=arr('M215 94 V172',k,2,id=a)+arr('M280 199 H328',k,2,id=a)+arr('M400 234 V296',k,2,id=a)+arr('M460 296 V236',k,2,id=a)+arr('M520 199 H588',k,2,id=a)+arr('M665 172 V96',k,2,id=a)
S.append(('5 · 暖色纸感','奶油底、赭红 / 芥黄 / 橄榄绿，胶囊形节点',svg(5,bg,b,k)))

# 6 单色 + 强调
b=''
b+=txt(0,60,'API server','#94a0ad',12,'start',600)+txt(0,190,'EngineCore','#94a0ad',12,'start',600)+txt(0,320,'GPU','#94a0ad',12,'start',600)
b+=f'<line x1="0" y1="118" x2="760" y2="118" stroke="#e3e7ec"/><line x1="0" y1="256" x2="760" y2="256" stroke="#e3e7ec"/>'
g='#3d4652';l='#c9d0d8'
def circ(x,y,r,fill,t,tc='#fff',fs=13):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'+txt(x,y,t,tc,fs,'middle',600)
b+=circ(180,60,40,l,'tokenize','#333',12)+circ(600,60,40,l,'detokenize','#333',11)
b+=circ(180,190,40,l,'输入线程','#333',12)+circ(400,190,58,'#ff5a1f','主线程',fs=16)+circ(600,190,40,l,'输出线程','#333',12)+circ(400,320,40,g,'GPU','#fff',14)
k='#ff5a1f';a='a6'
b+=arr('M180 100 V148',k,2.5,id=a)+arr('M218 190 H340',k,2.5,id=a)+arr('M385 246 V284',k,2.5,id=a)+arr('M415 282 V246',k,2.5,id=a)+arr('M458 190 H560',k,2.5,id=a)+arr('M600 150 V102',k,2.5,id=a)
b+=txt(188,128,'ZMQ','#94a0ad',11,'start')+txt(608,128,'ZMQ','#94a0ad',11,'start')
S.append(('6 · 单色 + 强调','灰色为底，只有关键路径与主线程用橙色，圆形节点',svg(6,'#fff',b,k)))

# 7 便当格 大色块
b=''
def tile(x,y,w,h,fill,t,sub=None,tc='#fff',fs=20):
    o=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}"/>'+txt(x+16,y+26,t,tc,fs,'start',700)
    if sub:o+=txt(x+16,y+h-22,sub,tc,12,'start',400)
    return o
b+=tile(8,8,360,160,'#0f9d8a','API server','tokenize · detokenize · 网络 IO，不占主线程')
b+=tile(376,8,376,160,'#e8ecf1','输入 / 输出线程','ZMQ socket ⇄ 队列，释放 GIL','#222')
b+=tile(8,176,520,196,'#7a3ff2','EngineCore 主线程','schedule → 准备输入 → 提交 → update')
b+=tile(536,176,216,196,'#2f6bff','GPU','forward + sample')
k='#fff';a='a7'
b+=arr('M300 120 H400',k,3,id=a)+arr('M560 120 V190',k,3,id=a)
b+=arr('M480 300 H556',k,3,id=a)+arr('M556 336 H480',k,3,id=a)
b+=txt(340,106,'请求','#fff',12)+txt(340,140,'','#fff')
S.append(('7 · 便当格','大面积色块 + 大字，面积表示重要性，箭头极简',svg(7,'#fff',b,k,380)))

# 8 地铁线路
b=''
k='#7a3ff2';a='a8'
pts=[(90,190,'tokenize','#0f9d8a'),(230,190,'输入线程','#8a94a3'),(400,190,'主线程','#7a3ff2'),(570,190,'输出线程','#8a94a3'),(670,190,'detokenize','#0f9d8a')]
b+=f'<path d="M90 190 H670" stroke="#7a3ff2" stroke-width="8" stroke-linecap="round"/>'
b+=f'<path d="M400 190 V320" stroke="#2f6bff" stroke-width="8" stroke-linecap="round"/>'
for x,y,t,cl in pts:
    b+=f'<circle cx="{x}" cy="{y}" r="15" fill="#fff" stroke="#24282b" stroke-width="4"/>'+txt(x,y-42,t,'#24282b',14,'middle',700)
b+=f'<circle cx="400" cy="320" r="22" fill="#2f6bff"/>'+txt(400,320,'GPU','#fff',13,'middle',700)+txt(400,364,'forward + sample','#2f6bff',13,'middle',600)
b+=txt(160,220,'ZMQ','#666',11)+txt(620,220,'ZMQ','#666',11)+txt(320,220,'入队','#666',11)+txt(485,220,'出队','#666',11)
b+=txt(0,20,'API server','#0f9d8a',12,'start',700)+txt(0,40,'EngineCore（进程 2）','#7a3ff2',12,'start',700)
b+=txt(412,254,'提交 batch ↓ ↑ 采样结果','#2f6bff',12,'start',600)
S.append(('8 · 线路图','一条主线串起各站，GPU 为支线站点',svg(8,'#fff',b,k)))

h='<!doctype html><meta charset=utf-8><title>engine process path · 8 styles</title><body style="margin:24px;background:#f4f5f7;font-family:-apple-system,\'PingFang SC\',sans-serif"><h2>“一个请求穿过的进程与线程”· 8 种风格</h2><div style="display:grid;grid-template-columns:1fr 1fr;gap:22px">'
for t,d,s in S:
    h+=f'<div style="background:#fff;padding:12px;border-radius:12px"><b>{t}</b> <span style="color:#777;font-size:13px">{d}</span>{s}</div>'
open('engine-process-styles.html','w').write(h+'</div>')
