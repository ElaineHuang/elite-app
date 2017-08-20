const SAVE_FORM = '/api/save_form';


$( "#analysisForm" ).submit(function(event) {
  event.preventDefault();
  
  if ($("#action-code option:selected").text() === "others" && $("#remark").val() === "") {
    alert('Remark field is required!!');
    return;
  }

  const formData = standardObject($(this).serializeArray());

  $(this)[0].reset();

  console.log(formData)

  callApi(SAVE_FORM, 'POST', formData, () => {
    alert('We received your data, thanks!');
    $('#actionTable tbody tr').remove();
    updateTable();
  });
});

function standardObject(orginialArray) {
  const result = {};

  orginialArray.forEach((field) => {
    result[field.name] = field.value;
  });

  return result;
};


$('#event-date').fdatepicker({
  format: 'yyyy/mm/dd hh:00',
  disableDblClickSelection: true,
  pickTime: 1,
  minView: 1,
  leftArrow:'<<',
  rightArrow:'>>'
});


