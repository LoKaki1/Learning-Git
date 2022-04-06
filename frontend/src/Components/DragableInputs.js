import { useState } from "react"
import Draggable from "react-draggable"
import { Switch } from "@mui/material"
export default function DraggableInputs (props) {
    const placeholder = `Enter ${props.text}...`
    const [dragPos, setDragPos] = useState({
        'x': props.defaultX,
        'y':props.defaultY
    })
    const [pos, setPos] = useState({
        'x': props.defaultX,
        'y':props.defaultY
    })
    async function handleDrag(e){
        setDragPos({
            'x': e.clientX,
            'y':e.clientY
        })
        console.log({
            'x': e.clientX,
            'y':e.clientY
        })
    }
    return(
        <>
        <div style={{position:'absolute', left: 1620, top:50}}>
          <Switch onClick={() => {
            props.setIsDraggable(!props.isDraggable)
            console.log(props.isDraggable)
            setPos(dragPos)
            console.log(pos)
          }}> 
          </Switch>
        </div> 
        {props.isDraggable ? 
            
            <Draggable positionOffset={pos} onDrag={e => handleDrag(e)} >
                <input  className='parameters'   type="text" onChange={(e) => props.func(e, props.text)} placeholder={placeholder} />   
            </Draggable>
            
         : 
         <div style={{position:'absolute', top:pos.y, left:pos.x}}>
            <input className='parameters'   type="text" onChange={(e) => props.func(e, props.text)} placeholder={placeholder} />   
         </div>
        }
        </>
    )
}