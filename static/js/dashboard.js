const GET_DANDGER_ERROR_CODE = '/api/get_danger_error_code';

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