const GET_ERROR_CODE = '/api/get_error_code';
const GET_NEW_RECORD = '/api/get_new_record';

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

callApi(GET_ERROR_CODE, 'GET', null, (errorCode) => {
  console.log(errorCode);
});

const updateTable = () => {
  callApi(GET_NEW_RECORD, 'GET', null, (result) => {
    const actionCodeList = JSON.parse(result.data);
    console.log(actionCodeList);

    let tableChild = "";

    for (let i = 0; i < actionCodeList.length; i++) {
      const actionCode = actionCodeList[i];
      tableChild = tableChild + `<tr><td>${actionCode.machine}</td><td>${actionCode.light}</td><td>${actionCode.actionCode}</td><td>${actionCode.remark}</td><td>${actionCode['event-date']}</td><td>${actionCode['update_time']}</td></tr>`;
    }

    $("#actionTable > tbody").append(tableChild);
  });
}

updateTable();