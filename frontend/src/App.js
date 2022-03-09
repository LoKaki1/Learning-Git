import React, {useState} from 'react';
import ParametersInput from './Components/Parameters.js';
import Watchlist from './Components/Watchlist.js';

export default function App() {
  
  const [benny, setBenny] = useState({}) 
  const [brendon, setBrendon] = useState([])
    return (
    <>
      <ParametersInput active={benny}  all={brendon} setAll={setBrendon} setActive={setBenny}/>
      <Watchlist rowsProp={brendon} active={benny} setAll={setBrendon}/>
    </>
    )
  
  }

