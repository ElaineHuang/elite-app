const GET_DANDGER_ERROR_CODE = '/api/get_danger_error_code';
const GET_HEALTH_INDEX = '/api/get_health_index';

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

  callApi(GET_HEALTH_INDEX, 'GET', null, (result) => {
    const data = result.data;

    $("#M023-health").html(data.M023 || 73.0);
    $("#M024-health").html(data.M024 || 15.0);
    $("#M025-health").html(data.M025 || 23.0);
<<<<<<< Updated upstream
    $("#M026-health").html(data.M026 || 68);
=======
    $("#M026-health").html(data.M026 || 68.0);
>>>>>>> Stashed changes
    
    $('#M023-ping').css('transform', 'rotate(' + convertDeg(data.M023 || 73.0) +')');
    $('#M024-ping').css('transform', 'rotate(' + convertDeg(data.M024 || 15.0) +')');
    $('#M025-ping').css('transform', 'rotate(' + convertDeg(data.M025 || 23.0) +')');
    $('#M026-ping').css('transform', 'rotate(' + convertDeg(data.M026 || 68.0) +')');
  });
});
