
import React, {useState} from 'react';
import ClientPage from './Components/clientPage';
import {BrowserRouter as Router, Route, Routes } from "react-router-dom";


import RegisterPage from './Components/registerPage';

export default function App() {
  const [clentToken, setClientToken] = useState()

  return (
    <>
      <Router>
        <Routes>
          <Route path='/' element={<RegisterPage/>}/>
          <Route path='/home' element={<ClientPage/>}></Route>
        </Routes>
      </Router>
    </>
  )
  
  }

