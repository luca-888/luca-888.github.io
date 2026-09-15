declare module '*.md' {
  const document: {
    title: string;
    html: string;
  };
  export default document;
}
