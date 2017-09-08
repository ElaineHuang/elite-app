const SAVE_FORM = '/api/save_form';


$( "#analysisForm" ).submit(function(event) {
  event.preventDefault();
  
  if ($("#action-code option:selected").text() === "A9 其它" && $("#remark").val() === "") {
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

$( "#upload-form" ).submit(function(event) {
  if ($("#machine").val() === "") {
    alert("請填入機台！");
    return false;
  }
});

function standardObject(orginialArray) {
  const result = {};

  orginialArray.forEach((field) => {
    result[field.name] = field.value;
  });

  return result;
};

if ($('#event-date')[0]) {
  $('#event-date').fdatepicker({
    format: 'yyyy/mm/dd hh:ii',
    disableDblClickSelection: true,
    pickTime: 1,
    // minView: 2,
    leftArrow:'<<',
    rightArrow:'>>'
  });
}
