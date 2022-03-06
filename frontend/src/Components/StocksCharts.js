import Chart from 'react-apexcharts'

export default function StocksCharts(props){

    const options =  {
        chart: {
          type: 'candlestick',

          width: '120%',
          foreColor: '#00',
          background: 'transpent'
        },
        title: {
          text: 'CandleStick Chart',
          align: 'left'
        },
        xaxis: {
          type: 'datetime'
        },
        yaxis: {
          tooltip: {
            enabled: true
          }
        }
    }

    return (
        <Chart className='graph'options={options} series={props.data} type="candlestick" width={500} height={320} />
    )
      
}

