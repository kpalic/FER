import React from "react";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'

import './App.css'

import Header from './components/Header/Header'
import Footer from './components/Footer/Footer'
import HomePage from "./pages/HomePage/HomePage";
import Map from "./pages/Map/Map";
import { useAuth } from "./auth/authContext";
import Login from "./pages/Login/Login";
import RegisterOwner from "./pages/Register/RegisterOwner";
import Owner from "./pages/ProfilePage/Owner";
import User from "./pages/ProfilePage/User";
import RegisterUser from "./pages/Register/RegisterUser";
import Register from "./pages/Register/Register";
import SuccessfulRegistration from "./pages/SuccessfulRegistration/SuccessfulRegistration";
import UserEdit from "./pages/ProfilePage/UserEdit";
import OwnerEdit from "./pages/ProfilePage/OwnerEdit";


function App(){

    const auth=useAuth()
    const [loadingRefresh, setLoadingRefresh] = React.useState<boolean>(true)

    React.useEffect(()=>{
        if(!auth.user){
            if(!auth.loading){
                setLoadingRefresh(false)
            }
        }else{
            setLoadingRefresh(false)
        }
    })
    return(
        <div className="App">
            <Router>
            <Header />
            <main>
                {!loadingRefresh ? (
                    <Routes>
                        <Route path="/" element={<HomePage/>}/>
                        <Route path="*" element={<HomePage/>}/>

                        {sessionStorage.getItem("token")?(
                            //auth.user?.accessToken?
                            //ako ima token
                            <>
                                <Route path="profile/owner" element={<Owner/>}/>
                                <Route path="profile/user" element={<User/>}/>
                                <Route path="profile/owner/edit" element={<OwnerEdit/>}/>
                                <Route path="profile/user/edit" element={<UserEdit/>}/>

                            </>
                        ):(
                            //ako token ne postoji
                            <>
                                <Route path="auth/register" element={<Register/>}/>
                                <Route path="auth/login" element={<Login />}/>
                                <Route path="auth/register/owner" element={<RegisterOwner />}/>
                                <Route path="auth/register/user" element={<RegisterUser />}/>
                                <Route path="auth/email-confirm" element={<SuccessfulRegistration />}/>
                            </>
                        )}
                         <Route path="map" element={<Map/>}/>
                    </Routes>
                ) : (
                    <></>
                )}

            </main>
            <Footer />
            </Router>
        </div>
    )
}

export default App