$(document).ready(function() {
    const recChart = c3.generate({
        bindto: '#rec-chart',
        data: {
            x : 'x',
            columns: [
                ['x', ' E530020', ' E530021', ' E530023', ' E530024', ' E530025', ' E530026'],
                ['February', 30, 200, 100, 400, 150, 250],
            ],
            types: {
                February: 'bar',
            }
        },
        axis: {
            x: {
                type: 'category',
                tick: {
                    multiline: false
                }
            },
            rotated: true
        }
    });
});