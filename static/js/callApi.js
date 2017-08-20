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

callApi(GET_NEW_RECORD, 'GET', null, (result) => {
  console.log(result);
});