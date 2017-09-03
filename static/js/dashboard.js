const GET_DANDGER_ERROR_CODE = '/api/get_danger_error_code';

const convertDeg = (number) => {
  return 180 * number / 100 + 'deg';
}

$(document).ready(() => {
  callApi(GET_DANDGER_ERROR_CODE, 'GET', null, (result) => {
    const machineList = result.data;

    let tableChild = '';

    $.each(machineList, (machine, errorCodes) => {
      tableChild = tableChild + `<tr><td>${machine}</td>`;
      $.each(errorCodes, (code, timeList) => {
        tableChild = tableChild + `<td class='align-center'>${timeList.length > 0 ? '<span class="error-icon"></span>' : '<span class="success-icon"></span>'}</td>`;
      });
      tableChild = tableChild + '</tr>'; 
    });

    $("#errorCodeTable > tbody").append(tableChild);
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
