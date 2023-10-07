import './Header.css';
import { useState } from "react";
import { UserRole } from "../../models/User.model";
import { useAuth } from "../../auth/authContext";

const Header = () => {

    const auth = useAuth()

    const [hamburger_class, setHamburgerClass] = useState("hamburger-bar unclicked")
    const [menu_class, setMenuClass] = useState("menu hidden")
    const [isMenuClicked, setIsMenuClickled] = useState(false)
    const [userRole, setUserRole] = useState("")

    const updateMenu = () => {
        if (!isMenuClicked) {
            setHamburgerClass("hamburger-bar clicked")
            if (sessionStorage.getItem("token")) {
                setMenuClass("menu2 visible")
            } else {
                setMenuClass("menu visible")
            }
        } else {
            setHamburgerClass("hamburger-bar unclicked")
            setMenuClass("menu2 hidden")
            setMenuClass("menu hidden")
        }
        setIsMenuClickled(!isMenuClicked)
    }

    const updateRole = () => {
        if (JSON.parse(sessionStorage.getItem('role') as string) === UserRole.ROLE_OWNER) {
            setUserRole("owner")
        } else {
            setUserRole("user")
        }
    }

    return (
        <div className="header">
            <div className="header-logo">
                <a href='/' className='header-logo-link'>
                    <img src='../../../logo.svg' alt="Dog Friendly Logo" className='logo'></img>
                </a>
            </div>

            <div className="navbar">
                <div className="navbar-item">
                    <a href='/' className='navbar-item-link'>Početna</a>
                </div>

                <div className="navbar-item">
                    <a href='/map' className='navbar-item-link'>Karta</a>
                </div>

                <div className="navbar-item" onClick={updateRole}>
                    {sessionStorage.getItem("token") ? (
                        <a href={"/profile/" + userRole } className='navbar-item-link'>Profil</a>
                    ):(
                        <a href="/auth/login" className='navbar-item-link'>Prijava</a>
                    )}
                </div>

                <div className="navbar-item" onClick={auth.logout}>
                    {sessionStorage.getItem("token") ? (
                        <a href="/" className='navbar-item-link'>Odjava</a>
                    ):(
                        <></>
                    )}
                </div>
            </div>

            <div className="hamburger">
                <nav className="hamburger-container">
                    <div className="hamburger-menu" onClick={updateMenu}>
                        <div className={hamburger_class} onClick={updateMenu}></div>
                        <div className={hamburger_class} onClick={updateMenu}></div>
                        <div className={hamburger_class} onClick={updateMenu}></div>
                    </div>
                </nav>

                <div className={menu_class}>
                    <div className="menu-dropdown" >
                        <div><a className="menu-item-link" href="/">Početna</a></div>
                        <div><a className="menu-item-link" href="/map">Karta</a></div>
                        {sessionStorage.getItem("token") ? (
                            <div onClick={updateRole}><a className="menu-item-link" href={"/profile/" + userRole}>Profil</a></div>
                        ):(
                            <div><a className="menu-item-link" href="/auth/login">Prijava</a></div>
                        )}
                        {sessionStorage.getItem("token") ? (
                            <div onClick={auth.logout}><a className="menu-item-link" href="/">Odjava</a></div>
                        ):(
                            <></>
                        )}
                    </div>
                </div>
            </div>
        </div>
)}
export default Header