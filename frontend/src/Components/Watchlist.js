import { DataGrid } from '@mui/x-data-grid'

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

    return (
        <div className='data-grid'>
            <DataGrid
             columns={columns}
             rows={rows} 
             onCellClick={
                 (params, t) => {
                     if (!t.ctrlKey){
                         console.log(params.row.ticker)
                         // Here display wha2t you want about the function using whatever shit you want (graph, data, kill me, and more :))
                     }
                 }
                
             }/>
        </div>
    )
}