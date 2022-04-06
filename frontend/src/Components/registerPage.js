import React from "react";
import '../Common/StyleSheets/LoginReg.css'

export default function RegisterPage() {
    function MyText(props){
        console.log(props)
        return (
            <div className="form-text-noder">
                <input type={`${props.placeholder}`} className="form-control" placeholder={`${props.placeholder}..`}/>
            </div>
        )        
    }
    return (
        <div  style={{width: 400,  backgroundColor:'rgb(34, 64, 78)', height: 350, left:'35%', top:'20%', position:'relative' }} className='main-div'>   
        <form >
            <h3 style={{textAlign: "center", color: 'white', flex: 1}}>Sign Up</h3>

            <MyText type='text' placeholder='First name'/>
            <MyText type='text' placeholder='Last name'/>
            <MyText type='email' placeholder='Enter email'/>
            <MyText type='password' placeholder='Enter password'/>
    
            <button style={{position:'absolute', left: '9%', marginTop: 10, paddingTop: 10}} type="submit" className="btn btn-primary btn-block">Sign Up</button>
            <p className="forgot-password text-right">
                Already registered <a href="#">sign in?</a>
            </p>
        </form>
        </div> 
    );
}
