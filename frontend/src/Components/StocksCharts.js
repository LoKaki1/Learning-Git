
import Chart from 'react-apexcharts'


export default function StocksCharts(props){

    const options =  {
        chart: {
          type: 'candlestick',
          foreColor: '#fff',
          background: 'rgb(34, 64, 78)',
          borderRadius: '100px',
        },
        plotOptions: {
          candlestick: {
            colors: {
              upward: 'rgba(25, 224, 158, 1)',
              downward: 'rgba(216, 80, 80, 1)'
            },
            wick: {
              useFillColor: true
            }
          }
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
      <>
        <Chart className='graph'  options={options} series={props.data} type="candlestick"  width={520} height={550} />
      </>
    )
      
}

