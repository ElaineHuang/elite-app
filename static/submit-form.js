$( "#analysisForm" ).submit(function(event) {
  event.preventDefault();
  const formData = standardObject($(this).serializeArray());
  $(this)[0].reset();

  console.log(formData)
});

function standardObject(orginialArray) {
  const result = {};

  orginialArray.forEach((field) => {
    result[field.name] = field.value;
  });

  return result;
};
