$(document).ready(function() {
    const recChart = c3.generate({
        bindto: '#rec-chart',
        size: {
            height: 800,
        },
        data: {
            x : 'x',
            columns: [
                ['x', 'E530020L', 'E551010L', 'E551010R', 'E551092R', 'E551502R', 'E551506R', 'E600003L', 'E600003R', 'E600102R', 'E605179R', 'E620002L', 'E620002R', 'E620003L', 'E620003R', 'E632102L', 'E632103L', 'E632104L', 'E632201L', 'E632236L', 'E632436R', 'E632501R'],
                ['February', 2, 28, 21, 1, 3, 3, 3, 1, 1, 1, 71, 70, 17, 18, 1, 1, 3, 3, 1, 1, 5],
            ],
            types: {
                February: 'bar',
            },
            colors: {
                February: '#ff6666'
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