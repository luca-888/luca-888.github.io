// 把渲染好的文章块按真实高度装进 1080 × 1440 的页面。
// 规则：标题不落在页尾（与下一块同页）；段落按句、列表按项、代码按行拆分，拆出的片段至少两行；
// 图表不拆，放不下就换页，单页仍放不下时整体缩小。

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

export function paginate(source: HTMLElement, container: HTMLElement, headerTitle: string): XhsResult {
  const pages: { page: HTMLElement; content: HTMLElement }[] = []
  const shrunk: XhsResult['shrunk'] = []
  let content!: HTMLElement

  const newPage = () => {
    if (pages.length >= 200) throw new Error('分页超过 200 页，可能陷入循环')
    const page = document.createElement('section')
    page.className = 'xhs-page'
    page.innerHTML = `<header><span></span><span>luca’s blog</span></header><div class="xhs-content xhs-body"></div><footer><span></span><span></span></footer>`
    page.querySelector('header span')!.textContent = headerTitle
    container.appendChild(page)
    content = page.querySelector('.xhs-content')!
    pages.push({ page, content })
  }
  const fits = () => content.scrollHeight <= content.clientHeight + 1

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

  // 不可拆的块（图、表、公式）缩小到当前页剩余高度；缩得太多则返回 false。
  const shrinkToFit = (el: HTMLElement, limit: number) => {
    content.appendChild(el)
    const overflow = content.scrollHeight - content.clientHeight
    const scale = (el.offsetHeight - overflow - 4) / el.offsetHeight
    if (scale < limit) { content.removeChild(el); return false }
    el.style.zoom = String(scale)
    shrunk.push({ page: pages.length, scale: Math.round(scale * 100) / 100 })
    return true
  }

  const place = (el: HTMLElement) => {
    content.appendChild(el)
    if (fits()) return
    content.removeChild(el)

    // 先用本块的前半部分填满当前页（标题与引导句因此能和正文同页），或把差一点放下的图缩小。
    if (content.childElementCount) {
      const result = split(el)
      if (result === 'all') return
      if (result === 'partial') { newPage(); place(el); return }
      if (!atomize(el) && shrinkToFit(el, SHRINK_LIMIT)) return
    }

    // 换页；页尾需要与下一块同页的元素一起带走。
    const carry: Element[] = []
    while (content.lastElementChild && keepWithNext(content.lastElementChild)) carry.unshift(content.removeChild(content.lastElementChild))
    newPage()
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
  newPage()
  for (const el of [...source.children] as HTMLElement[]) place(el)

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
