import { DataGrid } from '@mui/x-data-grid'
import StocksCharts from './StocksCharts';
import axios from 'axios';
import { useState } from 'react';

const columns = [
    { field: "ticker", headerName: "Ticker", renderCell: (cellValues) => {
        return (
        <div style={{              
        width: "100%",
        fontSize: 16, 
        }}>
        {cellValues.value}
        </div>)
    }},
    { field: "price", headerName: "Predicted Price", width: 130, renderCell: (cellValues) => {
        return (
          <div
            style={{
              color: "red" ,
              width: "100%",
              fontSize: 16, 
              textAlign: "center"
            }}
          >
            {cellValues.value}
          </div>
        );
    }},
    { field: "current_price", headerName: "Last Price",  renderCell: (cellValues) => {
        return (
          <div
            style={{
              color: "green" ,
              fontSize: 16,
              width: "100%",
              textAlign: "center"
            }}
          >
            {cellValues.value}
          </div>
        );
    }},
]


export default function Watchlist(props){
    const rows = props.rowsProp
    
    const [data, setData] = useState([])
    const graphStatic = <StocksCharts data={data}/>
    const [graph, setGraph] = useState(false, [])
    const getHistroicalData = (ticker) => {
      axios.post('http://127.0.0.1:5000/prices', {
        "ticker": ticker
      }).then((response) => {
        console.log(response)
        setData(() => {
          return [{
            'data': response.data
          }]
        })
      }, (error) => {
        console.log(error)
      })
    }
    return (
        <div className='data-grid'>
            <DataGrid
             columns={columns}
             rows={rows} 
             onCellClick={
                 (params, t) => {
                     if (!t.ctrlKey){
                        console.log(params.row.ticker)
                        getHistroicalData(params.row.ticker)
                        setGraph(true)
                     }
                 }
             }/>
             {graph ? graphStatic: <div/>}
             
        </div>
    )
}