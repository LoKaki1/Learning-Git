import { useState } from "react"
import Draggable from "react-draggable"

export default function DraggableInputs (props) {
    const placeholder = `Enter ${props.text}...`
    const [dragPos, setDragPos] = useState({
        'x': props.defaultX,
        'y':props.defaultY
    })
    
    function handleDrag(e){
        setDragPos({
            'x': e.clientX,
            'y':e.clientY
        })
    }
    return(
        <>
        {props.isDraggable ? <div  style={{position:'absolute', top:dragPos.y, left:dragPos.x}}>
            <Draggable onDrag={e => handleDrag(e)} >
                <input  className='parameters'   type="text" onChange={(e) => props.func(e, props.text)} placeholder={placeholder} />   
            </Draggable>
        </div> : 
        <div style={{position:'absolute', top:dragPos.y, left:dragPos.x}}>
            <input  className='parameters' type="text" onChange={(e) => props.func(e, props.text)} placeholder={placeholder}/>  
        </div>}
        </>
    )
}