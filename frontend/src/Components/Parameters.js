import React, {useState, useEffect} from 'react';
import axios from 'axios';
import './components.css'
import { Switch } from '@mui/material';
import DraggableInputs from './DragableInputs';

export default function ParametersInput(props) {
    const [load, setLoad] = useState(false, [])  
    useEffect(() => {
      axios.get('http://127.0.0.1:5000/watchlist').
      then((response) => {
        props.setAll(response.data)
      }, (error) =>{
        console.log(error)
      })
    }, [])
    const loader = <div className='loader'/>
    const handleChange = (e, key) => {
      console.log(key)
      const input = e.target;
      props.setActive((oldBenny) => {
        return { 
        ...oldBenny,  
        [`${key}`]: input.value.toUpperCase()
      }
      
      })
      console.log(input.value.toUpperCase())
        
    }
    const predictStock = (active=props.active, setAll=props.setAll) => {
        setLoad(true)
        axios.post('http://127.0.0.1:5000/predict', active || {"ticker": 'NIO'})
        .then((response) => {
          console.log(response)
          setAll(response.data)
          setLoad(false)
        }, (error) => {
          console.log(error);
        })
    }
    return (
    <>
      <div className='divs'>
        <div>
          <button className='submit'onClick={() => predictStock(props.active, props.setAll)}>
              Predict 🚀   
          </button>        
        </div>  
        <form className="bar">
            <input className='searchbar' type="text" onChange={(e) => handleChange(e, 'ticker')} placeholder='Enter ticker..'/>
        </form>
        {load ? loader: <div></div>}
        </div>
        <div style={{position:'absolute', left: 1620, top:50}}>
          <Switch onClick={() => {
            props.setIsDraggable(!props.isDraggable)
            console.log(props.isDraggable)
          }}> 
          </Switch>
        </div>  
        <DraggableInputs func={handleChange} text={'epochs'} isDragable={props.isDragable} defaultX={1466} defaultY={109} setIsDraggable={props.setIsDraggable} isDraggable={props.isDraggable}/>
        <DraggableInputs func={handleChange} text={'units'} isDragable={props.isDragable} defaultX={1466} defaultY={159} setIsDraggable={props.setIsDraggable} isDraggable={props.isDraggable}/>
        <DraggableInputs func={handleChange} text={'prediction days'} isDragable={props.isDragable} defaultX={1466} defaultY={209} setIsDraggable={props.setIsDraggable} isDraggable={props.isDraggable}/>
        <DraggableInputs func={handleChange} text={'prediction day'} isDragable={props.isDragable} defaultX={1466} defaultY={259} setIsDraggable={props.setIsDraggable} isDraggable={props.isDraggable}/>

    
        </>
    )
}