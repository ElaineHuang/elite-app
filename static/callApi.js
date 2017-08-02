const GET_ERROR_CODE = '/api/get_error_code';

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
  data = JSON.parse(errorCode.data);
  
  M023 = data.filter((d) => d.Machine === 'M023');
  M024 = data.filter((d) => d.Machine === 'M024');
  M025 = data.filter((d) => d.Machine === 'M025');
  M026 = data.filter((d) => d.Machine === 'M026');

  const errorIds = data.map((d) => d.ErrCode);
  errorIds.unshift('x');
  const Repetitions23 = M023.map((d) => d.Repetition);
  Repetitions23.unshift('M023');

  const Repetitions24 = M024.map((d) => d.Repetition);
  Repetitions24.unshift('M024');

  const Repetitions25 = M025.map((d) => d.Repetition);
  Repetitions25.unshift('M025');

  const Repetitions26 = M026.map((d) => d.Repetition);
  Repetitions26.unshift('M026');

  const chart = c3.generate({
      bindto: '#rec-chart',
      size: {
          height: 4000,
      },
      data: {
          x : 'x',
          columns: [
              errorIds,
              Repetitions23,
              Repetitions24,
              Repetitions25,
              Repetitions26,
          ],
          type: 'bar',
          colors: {
            M023: '#4dc3ff',
            M024: '#0086b3',
            M025: '#33cccc',
            M026: '#248f8f',
          },
      },
      axis: {
          x: {
              type: 'category',
              tick: {
                  multiline: false
              }
          },
          rotated: true
      },
      regions: [
          { axis: 'y', start: 0, end: 10, class: 'normal' },
          { axis: 'y', start: 10, end: 50, class: 'high' },
          { axis: 'y', start: 50, end: 80, class: 'critical' },
      ]
  });
});
