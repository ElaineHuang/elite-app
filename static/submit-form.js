const SAVE_FORM = '/api/save_form';

$( "#analysisForm" ).submit(function(event) {
  event.preventDefault();
  const formData = standardObject($(this).serializeArray());
  $(this)[0].reset();

  callApi(SAVE_FORM, 'POST', formData, () => {
    alert('We received your data, thanks!');
  });
});

function standardObject(orginialArray) {
  const result = {};

  orginialArray.forEach((field) => {
    result[field.name] = field.value;
  });

  return result;
};
