import './UserEdit.css'
import Button from '@mui/material/Button';
import { Link } from 'react-router-dom'
import { useState, useEffect } from "react";
import { AxiosInstance } from '../../auth/AxiosInstance';
import { useFormik } from 'formik';
import * as Yup from 'yup'
import {AxiosResponse} from 'axios'
import { useNavigate } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { createTheme } from '@mui/material/styles'
import { UserRole } from "../../models/User.model"

const EditUserSchema=Yup.object().shape({
    username:Yup.string().required('Obavezno'),
    password:Yup.string().min(8, 'Lozinka mora imati minimalno 8 znakova'),
    passwordConfirmation: Yup.string()
    .test('passwords-match', 'Lozinke se ne podudaraju', function(value) {
      return this.parent.password === value
    })
})

const theme = createTheme({
    palette:{
        primary:{
            main:"#8fb4d6"
        }
    }
})

const UserEdit=()=>{
    
    const navigate = useNavigate()

    
    const [username, setUsername] = useState("")
    const [isError, setError] = useState("");
    const [isErrorVisible, setIsErrorVisible] = useState(false)
    
    
    useEffect(()=> {
        if (JSON.parse(sessionStorage.getItem('role') as string) === UserRole.ROLE_OWNER) {
            navigate("../../")
        } else {
            AxiosInstance.get("profile/user").then(res => {
                setUsername(res.data.username)
            })
        }
    }, []);

    const editUser = (
        username: string,
        password: string
    ) => {
      return new Promise<AxiosResponse>((resolve, reject) => {
        AxiosInstance.put('/profile/user/edit', {
          username,
          password
        })
          .then(res => {
            resolve(res.data)
          })
          .catch(err => {
            reject(err)})
      })
    }

    const formik = useFormik({
        enableReinitialize: true,
        initialValues: {
            username: username,
            password: '',
            passwordConfirmation: ''
        },
        validationSchema: EditUserSchema,
        onSubmit: values=> {
            editUser(values.username, values.password)
            .then(()=>{
                navigate('/profile/user')
            })
            .catch(err=>{
                setIsErrorVisible(true)
                setError(err.response.data.message);
            })
        }
    })


    return(
        <div className={isErrorVisible ? 'user-edit-error' : 'user-edit'}>
            
            {isErrorVisible ? 
                (<div className='user-error-container'>{isError}</div> ) : (<></>)
            }
            
            <form onSubmit={formik.handleSubmit}>
                <div className='row2'>
                    <div className='label-name'>
                        <label>
                            Korisničko ime:
                        </label>
                    </div>
                    
                    <div className='label-input'>
                    {!(formik.touched.username && formik.errors.username) ? 
                        (<input className='label'
                            id='username'
                            name="username" 
                            type="text"
                            onChange={formik.handleChange}
                            value={formik.values.username}/> 
                            ) : (<>
                        <input className='label-input-error'
                            id='username'
                            name="username" 
                            type="text"
                            onChange={formik.handleChange}
                            value={formik.values.username}
                        /> 
                        <span className='error'>{formik.errors.username}</span> </>)
                    }  
                    </div> 
                </div>

                <div className='row2'>
                    <div className='label-name'>
                        <label>
                            Lozinka:
                        </label>
                    </div>

                    <div className='label-input'>
                    {!(formik.touched.password && formik.errors.password) ? 
                        (<input className='label'
                            id='password'
                            name="password" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.password}/> 
                            ) : (<>
                        <input className='label-input-error'
                            id='password'
                            name="password" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.password}
                        /> 
                        <span className='error'>{formik.errors.password}</span></>)
                    }
                    </div>
                </div>

                <div className='row2'>
                    <div className='label-name'>
                        <label>
                            Ponovljena lozinka:
                        </label>
                    </div>

                    <div className='label-input'>
                    {!(formik.touched.passwordConfirmation && formik.errors.passwordConfirmation) ? 
                        (<input className='label'
                            id='passwordConfirmation'
                            name="passwordConfirmation" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.passwordConfirmation}/> 
                            ) : (<>
                        <input className='label-input-error'
                            id='passwordConfirmation'
                            name="passwordConfirmation" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.passwordConfirmation}
                        /> 
                        <span className='error'>{formik.errors.passwordConfirmation}</span></>)
                    }
                    </div>
                </div>

                <div className='buttons-user-edit-container'>
                    <div>
                        <ThemeProvider theme={theme}>
                            <Button
                            className='button3'
                            color="primary"
                            variant="contained"
                            type="submit"
                            size="large"
                            sx={{
                                '@media screen and (max-width: 720px)': {
                                    fontSize: 12,
                                },
                                '@media screen and (max-height: 550px)': {
                                    fontSize: 10,
                                },
                                '@media screen and (max-width: 360px)': {
                                    fontSize: 8,
                                }
                            }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/save.png')} alt='save'/>Spremi</div></Button>
                        </ThemeProvider>
                    </div>

                    <div>
                        <Link to="../profile/user" style={{ textDecoration: 'none', color: 'black' }}>
                            <ThemeProvider theme={theme}>
                                <Button 
                                color="primary"
                                className='button3' 
                                variant="contained"
                                size="large"
                                sx={{
                                    '@media screen and (max-width: 720px)': {
                                        fontSize: 12,
                                    },
                                    '@media screen and (max-height: 550px)': {
                                        fontSize: 10,
                                    },
                                    '@media screen and (max-width: 360px)': {
                                        fontSize: 8,
                                    }
                                }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/cancel.png')} alt='cancel'/>Odustani</div></Button>
                            </ThemeProvider>
                        </Link>
                    </div>
                </div>
            </form>
        </div>
    )
}
export default UserEdit