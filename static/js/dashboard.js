const GET_DANDGER_ERROR_CODE = '/api/get_danger_error_code';

const convertDeg = (number) => {
  return 180 * number / 100 + 'deg';
}

$(document).ready(() => {
  callApi(GET_DANDGER_ERROR_CODE, 'GET', null, (result) => {
    const machineList = result.data;

    let tableChild = '';
    let errorChild = '<div class="alert alert-warning">';
    const errors = [];
    $.each(machineList, (machine, errorCodes) => {
      tableChild = tableChild + `<tr><td>${machine}</td>`;
      $.each(errorCodes, (code, timeList) => {
        tableChild = tableChild + `<td class='align-center'>${timeList.length > 0 ? '<span class="error-icon"></span>' : '<span class="success-icon"></span>'}</td>`;
        if (timeList.length > 0) {
          errors.push(code);
          errorChild = errorChild + `${machine} - ${code} 發生時間： ${timeList.join(', ')}<br>`;
        }
      });
      tableChild = tableChild + '</tr>';
    });
    errorChild = errorChild + '</div>';
    $("#errorCodeTable > tbody").append(tableChild);
    if (errors.length > 0) {
      $("#error-list").append(errorChild);
    }
  });

  $("#M023-health").html("25");
  $("#M024-health").html("88");
  $("#M025-health").html("77");
  $("#M026-health").html("17");

  $('#M023-ping').css('transform', 'rotate(' + convertDeg(25) +')');
  $('#M024-ping').css('transform', 'rotate(' + convertDeg(88) +')');
  $('#M025-ping').css('transform', 'rotate(' + convertDeg(77) +')');
  $('#M026-ping').css('transform', 'rotate(' + convertDeg(17) +')');
});
