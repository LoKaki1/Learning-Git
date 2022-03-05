import React, {useState, useEffect} from 'react';
import axios from 'axios';
import './components.css'

export default function ParametersInput(props) {
    const [load, setLoad] = useState(false, [])  
    const loader = <div className='loader'/>
    const handleChange = (e, key) => {
      console.log(key)
      props.setActive((oldBenny) => {
        return { 
        ...oldBenny,  
        [`${key}`]: e.target.value
      }
      })
        
    }
    const predictStock = () => {
        setLoad(true)
        axios.post('http://127.0.0.1:5000/predict', props.active || {"ticker": 'NIO'})
        .then((response) => {
          console.log(response)
          props.setAll(response.data)
          setLoad(false)
        }, (error) => {
          console.log(error);
        })
        
        

    }
  
    return (
    <>
        <div className='divs'>
        <div>
                <button className='submit'onClick={() => predictStock()}>
                    Predict 🚀   
                </button>
                
            </div>
            
            <form className="bar">
                <input className='searchbar' type="text" onChange={(e) => handleChange(e, 'ticker')} placeholder='Enter ticker..' style={{}}/>
            </form>
            {load ? loader: <div></div>}
 
        </div>
        
        {/* <input type="text"  onChange={(e) => handleChange(e, 'epochs')} placeholder='Enter epochs..'/> 
            <input type="text"  onChange={(e) => handleChange(e, 'units')} placeholder='Enter units..'/> 
            <input type="text"  onChange={(e) => handleChange(e, 'prediction_days')} placeholder='Enter prediction days..'/> 
            <input type="text"  onChange={(e) => handleChange(e, 'predicition_day')} placeholder='Enter prediction day..'/>  */}
    </>
    )
}