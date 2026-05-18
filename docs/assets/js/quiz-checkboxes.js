// Make GFM task-list checkboxes clickable (kramdown renders them disabled).
// State is not persisted — quizzes are self-assessment only.
(function () {
  document.querySelectorAll('li > input[type="checkbox"][disabled]').forEach(function (cb) {
    cb.removeAttribute('disabled');
    cb.classList.add('quiz-option');
  });
})();
