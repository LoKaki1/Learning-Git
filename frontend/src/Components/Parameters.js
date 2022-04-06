import React, {useState, useEffect} from 'react'
import axios from 'axios'
import '../Common/StyleSheets/components.css'

export default function ParametersInput(props) {
    const [load, setLoad] = useState(false, [])  
    useEffect(() => {
      axios.post('http://127.0.0.1:5000/watchlist', {token: 'd8a712798e4589958a9be46e746944cc5657e50a'}).
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
        [`${key}`]: input.value.toUpperCase(),
        "token": 'd8a712798e4589958a9be46e746944cc5657e50a'
      }
      
      })
      console.log(input.value.toUpperCase())
        
    }
    const predictStock = (active=props.active, setAll=props.setAll) => {
        setLoad(true)
      
        axios.post('http://127.0.0.1:5000/predict', active || {"ticker": 'NIO', })
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
 
      <input  className='parameters'
              type="text"
              style={{position: 'absolute', left:1466, top:109}} 
              onChange={(e) => handleChange(e, 'epochs')} 
              placeholder='Enter epochs..'/>

    
        </>
    )
}