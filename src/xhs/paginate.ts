// 把渲染好的文章块按真实高度装进 1080 × 1920 的页面。
// 规则：按语义分组装页，标题、引导句与下一块同组，图与紧随其后的“上图……”说明同组，整组放不下就整组换页；
// 放不进一页时组内的图适度缩小；整组换页会留下大片空白时改为逐块排入；二级章节在本页剩余不足 35% 时才另起一页；
// 代码不拆，段落与列表只在自身较长或本页剩余空间较多时拆开；单块超过一页时才强制拆分或缩小。

export type XhsResult = {
  pages: number
  headings: { level: number; text: string; page: number }[]
  shrunk: { page: number; scale: number }[]
  textIntact: boolean
}

const MIN_PIECE = 130 // 拆分片段的最小高度，约两行正文
// 需要与下一块同页的块：标题，以及以冒号结尾的引导句。
const keepWithNext = (el: Element) => /^H[1-3]$/.test(el.tagName) || (el.tagName === 'P' && /[:：]\s*$/.test(el.textContent ?? ''))
const SHRINK_LIMIT = 0.78 // 不可拆的块差一点放不下时，缩小到这个比例以内留在本页
const FIGURE_SHRINK_LIMIT = 0.9 // 自绘图已按窄屏布局，缩小超过一成图内文字就偏小
// 块在本页允许缩小的下限：自绘图更严，表格与原图沿用 SHRINK_LIMIT。
const shrinkLimit = (el: HTMLElement) => el.classList.contains('fk-figure') ? FIGURE_SHRINK_LIMIT : SHRINK_LIMIT
const LIST_SPLIT_FREE = 0.3 // 本页剩余空间不少于这个比例时，列表才在本页拆开
const MAX_GAP = 0.45 // 整组换页会让本页留白超过这个比例时，改为逐块排入
const SECTION_MIN_FREE = 0.35 // 本页剩余不足此比例时，新的二级章节才另起一页
const PARA_SPLIT = 0.4 // 段落高于页面这个比例，或本页剩余空间高于这个比例时，才在本页拆开
const isFigure = (el: Element) => el.tagName === 'FIGURE' || el.classList.contains('table-scroll')
// 回指上一张图或表的说明段落，与图同组。
const refersBack = (el: Element) => el.tagName === 'P' && /^(上图|上面|图中|上表)/.test((el.textContent ?? '').trim())

export function paginate(source: HTMLElement, container: HTMLElement, headerTitle: string): XhsResult {
  const pages: { page: HTMLElement; content: HTMLElement }[] = []
  const shrunk: XhsResult['shrunk'] = []
  let content!: HTMLElement

  const newPage = () => {
    if (pages.length >= 200) throw new Error('分页超过 200 页，可能陷入循环')
    const page = document.createElement('section')
    page.className = 'xhs-page'
    page.innerHTML = `<header><span></span><span>流形笔记 · luca’s blog</span></header><div class="xhs-content xhs-body"></div><footer><span></span><span></span></footer>`
    page.querySelector('header span')!.textContent = headerTitle
    container.appendChild(page)
    content = page.querySelector('.xhs-content')!
    pages.push({ page, content })
  }
  const fits = () => content.scrollHeight <= content.clientHeight + 1
  const used = () => {
    const last = content.lastElementChild
    return last ? last.getBoundingClientRect().bottom - content.getBoundingClientRect().top : 0
  }

  // 把块拆成可逐个移动的原子：段落按句，列表按项，代码按行。
  const atomize = (el: HTMLElement): { part: HTMLElement; holder: HTMLElement; atoms: Node[] } | null => {
    if (el.tagName === 'P') {
      for (const node of [...el.childNodes]) {
        if (node.nodeType !== Node.TEXT_NODE) continue
        const pieces = node.textContent!.split(/(?<=[。；！？：])/)
        if (pieces.length > 1) node.replaceWith(...pieces.map(text => document.createTextNode(text)))
      }
      const part = el.cloneNode(false) as HTMLElement
      return { part, holder: part, atoms: [...el.childNodes] }
    }
    if (el.tagName === 'UL' || el.tagName === 'OL') {
      const part = el.cloneNode(false) as HTMLElement
      return { part, holder: part, atoms: [...el.children] }
    }
    if (el.tagName === 'PRE' && el.firstElementChild) {
      const part = el.cloneNode(false) as HTMLElement
      const code = el.firstElementChild.cloneNode(false) as HTMLElement
      part.appendChild(code)
      return { part, holder: code, atoms: [...el.firstElementChild.childNodes] }
    }
    return null
  }

  // 把块的前半部分装进当前页，剩余部分留在原块里。
  const split = (el: HTMLElement): 'none' | 'partial' | 'all' => {
    const parts = atomize(el)
    if (!parts) return 'none'
    const { part, holder, atoms } = parts
    content.appendChild(part)
    const origin = el.tagName === 'PRE' ? el.firstElementChild! : el
    let moved = 0
    for (const atom of atoms) {
      holder.appendChild(atom)
      if (!fits()) { origin.prepend(atom); break }
      moved++
    }
    if (moved === atoms.length) return 'all'
    // 剩余部分也不能太短：代码与列表至少留两项，段落至少留 12 个字。
    const rest = () => atoms.slice(moved)
    const tooShort = () => el.tagName === 'P'
      ? rest().reduce((n, atom) => n + (atom.textContent ?? '').trim().length, 0) < 12
      : rest().length < 2
    while (moved > 1 && tooShort()) { origin.prepend(atoms[--moved]); }
    if (moved === 0 || part.offsetHeight < MIN_PIECE) {
      origin.prepend(...holder.childNodes)
      part.remove()
      return 'none'
    }
    if (el.tagName === 'OL') (el as HTMLOListElement).start += moved
    return 'partial'
  }

  // 缩小一个块：自绘图按宽度缩 SVG（zoom 对自适应宽度的 SVG 无效），其余块用 zoom。
  const setScale = (el: HTMLElement, scale: number) => {
    const svg = el.classList.contains('fk-figure') ? el.querySelector<SVGElement>(':scope > svg') : null
    if (svg) { svg.style.width = scale === 1 ? '' : `${svg.getBoundingClientRect().width / (Number(svg.dataset.scale ?? 1)) * scale}px`; svg.dataset.scale = String(scale); svg.style.margin = '0 auto'; svg.style.display = 'block' }
    else el.style.zoom = scale === 1 ? '' : String(scale)
  }
  // 逐步缩小 target，直到当前页放得下；低于 limit 仍放不下则复原并返回 false。
  const shrinkUntilFits = (target: HTMLElement, limit: number) => {
    for (let scale = 0.98; scale >= limit - 1e-6; scale -= 0.02) {
      setScale(target, scale)
      if (fits()) { shrunk.push({ page: pages.length, scale: Math.round(scale * 100) / 100 }); return true }
    }
    setScale(target, 1)
    return false
  }

  // 不可拆的块（图、表、公式）缩小到当前页剩余高度；缩得太多则返回 false。
  const shrinkToFit = (el: HTMLElement, limit: number) => {
    content.appendChild(el)
    if (shrinkUntilFits(el, limit)) return true
    content.removeChild(el)
    return false
  }

  const place = (el: HTMLElement) => {
    content.appendChild(el)
    if (fits()) return
    const height = el.offsetHeight
    content.removeChild(el)

    // 先用本块的前半部分填满当前页，或把差一点放下的图缩小；代码不在页中拆开，列表只在剩余空间较多时拆。
    const free = content.clientHeight - used()
    const splitHere = (el.tagName === 'P' && (free >= content.clientHeight * PARA_SPLIT || height >= content.clientHeight * PARA_SPLIT))
      || ((el.tagName === 'UL' || el.tagName === 'OL') && free >= content.clientHeight * LIST_SPLIT_FREE)
    // 列表差一点放不下时，先把本页已有的图适度缩小，让整个列表留在本页，不拆开。
    const isList = el.tagName === 'UL' || el.tagName === 'OL'
    const figureHere = isList ? [...content.children].reverse().find(isFigure) as HTMLElement | undefined : undefined
    if (figureHere && Number(figureHere.querySelector<SVGElement>(':scope > svg')?.dataset.scale ?? 1) === 1 && !figureHere.style.zoom) {
      content.appendChild(el)
      if (shrinkUntilFits(figureHere, shrinkLimit(figureHere))) return
      content.removeChild(el)
    }
    if (content.childElementCount && splitHere) {
      const result = split(el)
      if (result === 'all') return
      if (result === 'partial') { newPage(); place(el); return }
    }
    if (content.childElementCount && !atomize(el) && shrinkToFit(el, shrinkLimit(el))) return

    // 换页；页尾需要与下一块同页的元素一起带走。
    // 本页已空（或只剩要带走的标题）时不再换页，避免留下空白页。
    const carry: Element[] = []
    while (content.lastElementChild && keepWithNext(content.lastElementChild)) carry.unshift(content.removeChild(content.lastElementChild))
    if (content.childElementCount) newPage()
    carry.forEach(node => content.appendChild(node))
    content.appendChild(el)
    if (fits()) return
    content.removeChild(el)
    const result = split(el)
    if (result === 'all') return
    if (result === 'partial') { newPage(); place(el); return }
    shrinkToFit(el, 0.4)
  }

  const plain = (text: string | null) => (text ?? '').replace(/\s+/g, '')
  const originalText = plain(source.textContent)
  // 一组连续的块：标题与引导句连同下一块，图连同回指它的说明。
  const blocks = [...source.children] as HTMLElement[]
  const groupEnd = (i: number) => {
    let j = i
    while (j + 1 < blocks.length && (keepWithNext(blocks[j]) || (isFigure(blocks[j]) && refersBack(blocks[j + 1])))) j++
    return j
  }
  const placeGroup = (group: HTMLElement[]) => {
    if (group[0].tagName === 'H2' && content.childElementCount && content.clientHeight - used() < content.clientHeight * SECTION_MIN_FREE) newPage()
    if (group.length > 1) {
      const tryHere = () => {
        group.forEach(el => content.appendChild(el))
        if (fits()) return true
        group.forEach(el => content.removeChild(el))
        return false
      }
      if (tryHere()) return
      if (content.childElementCount && content.clientHeight - used() <= content.clientHeight * MAX_GAP) {
        newPage()
        if (tryHere()) return
        // 图与说明放不进一页时，把组内的图缩小（不低于 SHRINK_LIMIT）留在同页。
        const figure = group.find(isFigure)
        if (figure) {
          group.forEach(el => content.appendChild(el))
          if (shrinkUntilFits(figure, shrinkLimit(figure))) return
          group.forEach(el => content.removeChild(el))
        }
      }
    }
    group.forEach(place)
  }

  newPage()
  for (let i = 0; i < blocks.length;) {
    const j = groupEnd(i)
    placeGroup(blocks.slice(i, j + 1))
    i = j + 1
  }

  // 页脚：当前章节（本页出现的第一个二级标题，否则沿用上一页）与页码。
  const headings: XhsResult['headings'] = []
  let chapter = ''
  pages.forEach(({ page, content }, index) => {
    const found = [...content.querySelectorAll('h1, h2, h3')] as HTMLElement[]
    found.forEach(h => headings.push({ level: Number(h.tagName[1]), text: h.textContent ?? '', page: index + 1 }))
    const h2 = found.find(h => h.tagName === 'H2')
    if (h2) chapter = h2.textContent ?? ''
    const spans = page.querySelectorAll('footer span')
    spans[0].textContent = chapter
    spans[1].textContent = `${index + 1} / ${pages.length}`
  })
  const pagedText = pages.map(({ content }) => plain(content.textContent)).join('')
  source.remove()
  return { pages: pages.length, headings, shrunk, textIntact: pagedText === originalText }
}
