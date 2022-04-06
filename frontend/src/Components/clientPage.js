import ParametersInput from "./Parameters"
import Watchlist from "./Watchlist"
import React, { useState } from "react"


export default function ClientPage(){
    const [benny, setBenny] = useState({}) 
    const [brendon, setBrendon] = useState([])
    const [isDraggable, setIsDraggable] = useState(true)
    return (
    <>

        <ParametersInput active={benny}  all={brendon} setAll={setBrendon} setActive={setBenny} isDraggable={isDraggable} setIsDraggable={setIsDraggable}/>
        <Watchlist rowsProp={brendon} active={benny} setAll={setBrendon}/>

    </>
    )
}