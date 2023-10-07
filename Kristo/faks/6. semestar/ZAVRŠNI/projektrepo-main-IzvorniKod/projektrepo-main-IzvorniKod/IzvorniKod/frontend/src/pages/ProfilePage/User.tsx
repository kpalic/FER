import { useState, useEffect} from 'react'
import { AxiosInstance } from '../../auth/AxiosInstance'
import { Link } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { createTheme } from '@mui/material/styles'
import { useNavigate } from 'react-router-dom'
import { useAuth } from "../../auth/authContext";
import { Ratings } from "../ProfilePage/Ratings"
import Button from '@mui/material/Button';
import { UserRole } from "../../models/User.model";
import './User.css'

const theme = createTheme({
    palette:{
        primary:{
            main:"#8fb4d6"
        }
    }
})

const User=()=> {

    const navigate = useNavigate()
    
    const [user, setUser] = useState("")
    const [loading, setLoading] = useState(true)
    const [data, setData] = useState<any>([])
    
    const auth = useAuth()
    
    useEffect(()=> {
        if (JSON.parse(sessionStorage.getItem('role') as string) === UserRole.ROLE_OWNER) {
            navigate("../../")
        } else {
            AxiosInstance.get("profile/user").then(res => {
                setUser(res.data)
                setData(res.data.ratings)
                setLoading(false)
            })
        }
    }, []);

    const deleteUserProfile = () => {
        AxiosInstance.delete('profile/user/delete', {
          }).then(() => {
            auth.logout()
            navigate('/')
            window.location.reload();
          }).catch(() => {
          })
    }

    return (
        <div className='user-page'>
            <div className='user'> 
                <div className='user-data-title'>
                    <h1>Korisnički podatci </h1>
                </div>

                <div className='user-data'>
                    <div className='row'>
                        <div className='data-name'>
                            Korisničko ime:
                        </div>
                        <div className='data'>
                            {(user as any).username}
                        </div>
                    </div>
                    <div className='row'>
                        <div className='data-name'>
                            Email:
                        </div>
                        <div className='data'>
                            {(user as any).email}
                        </div>
                    </div>
                    <div className='row'>
                        <div className='data-name'>
                            Lozinka:
                        </div>
                        <div className='data'>
                            ********
                        </div>
                    </div>
                </div>
            </div>

            <div className='buttons-container'>
                <div className='button-user-container'>
                    <Link to="edit" style={{ textDecoration: 'none', color: 'black' }}>
                    <ThemeProvider theme={theme}>
                        <Button
                        color="primary"
                        variant="contained"
                        sx={{
                            '@media screen and (max-width: 720px)': {
                                fontSize: 10,
                            },
                            '@media screen and (max-height: 550px)': {
                                fontSize: 10,
                            },
                            '@media screen and (max-width: 360px)': {
                                fontSize: 8,
                            }
                        }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/edit.png')} alt='edit'/>Uredi podatke</div></Button>
                        </ThemeProvider>
                        </Link>
                </div>

                <div className='button-user-container'>
                    <ThemeProvider theme={theme}>
                        <Button
                        color="primary"
                        variant="contained"
                        onClick={() => {
                            const confirmBox = window.confirm(
                                "Želite li zaista izbrisati svoj korisnički račun?"
                            )
                            if (confirmBox === true) {
                                deleteUserProfile()
                            }
                        }}
                        sx={{
                            '@media screen and (max-width: 720px)': {
                                fontSize: 10,
                            },
                            '@media screen and (max-height: 550px)': {
                                fontSize: 10,
                            },
                            '@media screen and (max-width: 360px)': {
                                fontSize: 8,
                            }
                        }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/delete2.png')} alt='delete'/>Izbriši korisnički račun</div></Button>
                        </ThemeProvider>
                </div>
            </div>

            {loading ? (<></>) : (<Ratings ratings = {data}/>)}
            
        </div>
    )
}
export default User