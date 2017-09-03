const GET_ERROR_CODE = '/api/get_error_code';
const GET_NEW_RECORD = '/api/get_new_record';
const GET_STATISTIC = '/api/get_statistic_data';

const callApi = (url = '', method = 'GET', data = null, callback) => {
  if (data) {
    data = JSON.stringify(data);
  }
  const client = new XMLHttpRequest();
  client.onreadystatechange = () => { 
    if (client.readyState === 4 && client.status === 200 && callback) {
      callback(JSON.parse(client.responseText));
    }
  };
  client.open(method, url, true); // true for asynchronous 
  client.setRequestHeader('Content-Type', 'application/json');
  client.send(data);
};

const updateTable = () => {
  callApi(GET_NEW_RECORD, 'GET', null, (result) => {
    const actionCodeList = JSON.parse(result.data);

    let tableChild = "";

    for (let i = 0; i < actionCodeList.length; i++) {
      const actionCode = actionCodeList[i];
      tableChild = tableChild + `<tr><td>${actionCode.machine}</td><td>${actionCode.light}</td><td>${actionCode.actionCode}</td><td>${actionCode.remark}</td><td>${actionCode['event-date']}</td><td>${actionCode['update_time']}</td></tr>`;
    }

    $("#actionTable > tbody").append(tableChild);
  });
}

const xLineGenerator = (errorCode) => {
  const machine = Object.keys(errorCode)[0];
  
  const xLine = ['x'];

  $.each(errorCode[machine], (year, months) => {
    $.each(months, (month) => {
      xLine.push(year+'-'+month);
    });
  });

  return xLine;
}

const machineListGenerator = (data) => {
  const machineList = {};
  // data = {"ErrorCode": {"M023": ..., "FiveMin"...
  $.each(data, (category, machineInfo) => {
    // category = ErrorCode, FiveMin
    // machineInfo = {"M023": ...}
    $.each(machineInfo, (index, value) => {
      const dataRow = [category];
      if (!machineList[index]) {
        machineList[index] = [];
      }

      $.each(value, (year, months) => {
        $.each(months, (mon, repetition) => {
          dataRow.push(repetition);
        });
      });

      // dataRow = ['ErrorCode', 23, 50 ...]
      machineList[index].push(dataRow);
    });
  });

  return machineList;
}

$(document).ready(() => {

  if (window.location.href.indexOf('statistic') > -1) {
    callApi(GET_STATISTIC, 'GET', null, (res) => {
      const data = res.data;
      console.log(data);
      let columns = [];

      const machineList = machineListGenerator(data);
      const machineNames = Object.keys(data.ErrorCode)
      const xLine = xLineGenerator(data.ErrorCode);

      columns.push(xLine);
      columns = columns.concat(machineList[$('input:radio[name=machine]:checked').val()]);

      const chart = c3.generate({
        bindto: '#error-code-chart',
        size: {
          width: 1170,
          height: 600,
        },
        data: {
          x: 'x',
          xFormat: '%Y-%m',
          columns: columns,
        },
        axis: {
          x: {
            type: 'timeseries',
            tick: {
              format: '%Y-%m'
            }
          }
        }
      });

      $('input:radio[name=machine]').change(function() {
        const checkedMachine = $('input:radio[name=machine]:checked').val();
        chart.load({
          columns: machineList[checkedMachine]
        });
      });
    });
  }

  if (window.location.href.indexOf('record') > -1) {
    updateTable();
  }

  const errorCodeList = [
    "E001001", "E031016", "E041205", "W940100", "W502001", "W432000", "W362000"
  ];

  errorCodeList.forEach((code) => {
    $('[data-toggle="' + code + '"]').popover();   
  });
});