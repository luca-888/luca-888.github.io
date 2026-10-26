import json, os, re, subprocess
OLD=os.path.join(os.path.dirname(os.path.abspath(__file__)),'old')
CSS='file:///Users/feihong/Documents/ChatGPT/luca-888.github.io/videos/matmul-ladder-tc-hf/assets/brand.css'
HERE=os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{HERE}/kfsrc', exist_ok=True); os.makedirs(f'{HERE}/img', exist_ok=True)
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
# narration
L={};cur=None
for l in open('/Users/feihong/Documents/ChatGPT/luca-888.github.io/videos/matmul-ladder-tc-hf/narration.md').read().splitlines():
    if l.startswith('## s'): cur=l[3:].strip(); L[cur]=[]
    elif l.startswith('## '): cur=None
    elif cur and l.strip(): L[cur].append(l.strip())
CH={'s03':'第一章：<span class="acc">读顺、送早</span>','s04':'第二章：<span class="acc">Hopper 的异步机制</span>','s05':'第三章：<span class="acc">少读、晚写</span>'}
# (新幕, 句号) -> (来源, chip, 说明)
M={
 's01':[('s01-01',None,'沿用'),('NEW_sm',None,'新画：SM 里画出寄存器、FP32 单元、Tensor Core、shared memory，Tensor Core 高亮'),('s01-03',None,'沿用'),('s01-04',None,'沿用'),('s01-05',None,'沿用'),('s01-06',None,'沿用')],
 's02':[('s02-01',None,'沿用'),('s02-02',None,'沿用'),('s02-03',None,'沿用'),('s02-04',None,'沿用')],
 's03':[('s03-01','mm_mma','沿用旧第 2 步画面，标题改为第一章'),('s03-02','mm_mma','沿用'),('s03-03','mm_mma','沿用'),('s03-04','mm_mma','沿用；cp.async 讲完耗时条缩到 4.09'),
        ('s04-02','mm_mma&lt;3级&gt;','沿用旧第 3 步的 stage 环与时间线；耗时条缩到 3.56'),('s04-03','mm_mma&lt;3级&gt;','沿用'),('s04-04','mm_mma&lt;3级&gt;','沿用；章末瓶颈')],
 's04':[('s05-01','mm_wgmma','沿用三个名字的画面'),('s05-02','mm_wgmma','沿用'),('s05-03','mm_wgmma','沿用'),('s05-04','mm_wgmma','沿用；耗时条缩到 2.27'),
        ('s06-01','mm_wgmma_ws','旧“同一组线程空着”的时间线删掉，直接接 producer/consumer 泳道'),('s06-02','mm_wgmma_ws','沿用 full / empty；两个 consumer 交替并入这一句'),('s06-04','mm_wgmma_ws','沿用；耗时条抖一下'),('s06-05','mm_wgmma_ws','沿用；章末瓶颈')],
 's05':[('s07-01','wgmma_l2','沿用旧第 6 步，标题改为第三章'),('s07-02','wgmma_l2','tile 0 / tile 32 与扇出合在这一句'),('s07-04','wgmma_l2','沿用；耗时条缩到 1.49'),
        ('s08-01','wgmma_cluster','沿用'),('s08-03','wgmma_cluster','两个 block 各搬一半与多播合在这一句；耗时条缩到 1.45'),('s08-04','wgmma_cluster','沿用；旧瓶颈标签去掉，直接接 TMA store'),
        ('s09-02','wgmma_tstore','沿用'),('s09-03','wgmma_tstore','预算条后接“1.41 ms · cuBLAS 的 99.5%”；耗时条缩到 1.41')],
 's06':[('s10-01',None,'沿用'),('s10-02',None,'沿用'),('NEW_lad',None,'新画：阶梯按三章分组，每章一个括号与章名'),('s10-04',None,'沿用')],
}
HEAD=f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{CSS}"><style>html,body{{margin:0;width:1920px;height:1080px;overflow:hidden;background:var(--paper);font-family:var(--zh)}}.a{{position:absolute;white-space:nowrap}}.sub{{position:absolute;left:0;right:0;bottom:56px;display:flex;justify-content:center}}.sub span{{max-width:1640px;white-space:normal;text-align:center;background:var(--sub-bg);color:#fff;font-size:34px;line-height:1.4;font-weight:600;padding:8px 24px;border-radius:4px}}</style></head><body>'
def sub(t): return f'<div class="sub"><span>{t}</span></div>'
def new_sm():
    parts=''
    for k,(x,y) in enumerate([(490,190),(970,190),(490,460),(970,460)]):
        parts+=f'<div class="a" style="left:{x}px;top:{y}px;width:460px;height:250px;background:var(--paper2);border:2px solid var(--line)">'
        parts+='<div class="a" style="left:20px;top:20px;width:190px;height:140px;background:var(--teal);color:#fff;font-size:34px;font-weight:800;display:flex;align-items:center;justify-content:center;text-align:center;line-height:40px;white-space:normal">Tensor<br>Core</div>'
        parts+='<div class="a" style="left:236px;top:20px;width:200px;height:96px;background-image:radial-gradient(circle,var(--ink2) 6px,transparent 6.5px);background-size:25px 24px"></div><div class="a" style="left:236px;top:120px;font-size:32px;font-weight:700;color:var(--ink2)">FP32 × 32</div>'
        parts+='<div class="a" style="left:20px;top:180px;width:420px;height:50px;line-height:50px;background:var(--purple);color:#fff;font-size:32px;font-weight:700;padding-left:14px;box-sizing:border-box">寄存器</div></div>'
    return f'<div class="a" style="left:460px;top:150px;width:1000px;height:740px;border:3px dashed var(--gray)"></div>{parts}<div class="a" style="left:490px;top:740px;width:940px;height:110px;line-height:110px;background:var(--use);color:#fff;font-size:40px;font-weight:800;text-align:center">shared memory</div>'
def new_lad():
    rows=[('第 1 版',11.78),('第 2 版',4.09),('第 3 版',3.56),('第 4 版',2.27),('第 5 版',2.29),('第 6 版',1.49),('第 7 版',1.45),('第 8 版',1.41)]
    h='<div class="a" style="left:96px;top:60px;font-size:36px;font-weight:700">8192 阶矩阵乘法 · 一次耗时</div>'
    for i,(v,ms) in enumerate(rows):
        y=140+i*88; w=min(ms,6)*68
        h+=f'<div class="a" style="left:96px;top:{y}px;font-size:32px;font-weight:700;color:{"var(--teal-t)" if i==7 else "var(--ink)"}">{v}</div><div class="a" style="left:240px;top:{y-4}px;width:{w:.0f}px;height:48px;background:{"var(--teal)" if i==7 else "var(--ink2)"}"></div><div class="a" style="left:{254+w:.0f}px;top:{y}px;font-family:var(--mono);font-size:32px;font-weight:800">{ms:.2f} ms</div>'
    for (a,b,name,sub_) in [(1,2,'第一章：读顺、送早','swizzle · cp.async · pipeline'),(3,4,'第二章：Hopper 的异步机制','TMA · wgmma · warp specialization'),(5,7,'第三章：少读、晚写','tile 发放顺序 · multicast · TMA store')]:
        y0=140+a*88-8; y1=140+b*88+52
        h+=f'<div class="a" style="left:800px;top:{y0}px;width:22px;height:{y1-y0}px;border:4px solid var(--teal-t);border-left:none"></div>'
        h+=f'<div class="a" style="left:860px;top:{(y0+y1)/2-46:.0f}px;font-size:40px;font-weight:800;color:var(--ink)">{name}</div><div class="a" style="left:860px;top:{(y0+y1)/2+6:.0f}px;font-family:var(--mono);font-size:32px;font-weight:700;color:var(--teal-t)">{sub_}</div>'
    h+=f'<div class="a" style="left:860px;top:{140-8+52/2-24}px;font-size:36px;font-weight:700;color:var(--ink2)">起点：wmma</div>'
    return h
frames=[]
for sc,lst in M.items():
    for n,(src,chip,note) in enumerate(lst,1):
        t=L[sc][n-1]; fid=f'{sc}-{n:02d}'
        if src=='NEW_sm': body=new_sm()
        elif src=='NEW_lad': body=new_lad()
        else:
            body=f'<img src="file://{OLD}/{src}.jpg" style="position:absolute;left:0;top:0;width:1920px;height:1080px">'
            if chip:
                body+=f'<div class="a" style="left:0;top:36px;width:1080px;height:196px;background:var(--paper)"></div><div class="kit-chip" style="top:60px">{chip}</div><div class="a" style="left:96px;top:122px;font-size:48px;font-weight:800">{CH[sc]}</div>'
            body+='<div class="a" style="left:0;top:920px;width:1920px;height:160px;background:var(--paper)"></div>'
        open(f'{HERE}/kfsrc/{fid}.html','w').write(HEAD+body+sub(t)+'</body></html>')
        subprocess.run([CHROME,'--headless=new','--disable-gpu','--hide-scrollbars','--allow-file-access-from-files','--window-size=1920,1080',f'--screenshot={HERE}/kfsrc/{fid}.png',f'file://{HERE}/kfsrc/{fid}.html'],capture_output=True)
        subprocess.run(['sips','-s','format','jpeg','-s','formatOptions','80','-Z','1280',f'{HERE}/kfsrc/{fid}.png','--out',f'{HERE}/img/{fid}.jpg'],capture_output=True)
        frames.append({'id':fid,'scene':sc,'n':n,'text':t,'img':f'img/{fid}.jpg','src':src,'note':note})
json.dump(frames,open(f'{HERE}/frames.json','w'),ensure_ascii=False)
print(len(frames))
