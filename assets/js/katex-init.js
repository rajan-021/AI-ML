// Render math written as $...$ / $$...$$ in Markdown. The arithmatex extension
// turns those into \( \) and \[ \] spans, so only those delimiters are scanned and
// dollar signs in normal prose are left alone.
window.AIML_MATH_OPTIONS = {
  delimiters: [
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true }
  ],
  throwOnError: false
};

document$.subscribe(function (doc) {
  if (window.renderMathInElement) {
    window.renderMathInElement(doc.body, window.AIML_MATH_OPTIONS);
  }
});
