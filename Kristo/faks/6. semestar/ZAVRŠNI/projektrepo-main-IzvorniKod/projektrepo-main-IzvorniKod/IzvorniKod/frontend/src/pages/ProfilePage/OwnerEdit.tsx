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
import { UserRole } from "../../models/User.model";

const EditOwnerSchema=Yup.object().shape({
    username:Yup.string().required('Obavezno'),
    password:Yup.string().min(8,'Lozinka mora imati minimalno 8 znakova'),
    passwordConfirmation: Yup.string().oneOf([Yup.ref('password'), null,], 'Lozinke se ne podudaraju')
    .test('passwords-match', 'Lozinke se ne podudaraju', function(value){
        return this.parent.password === value}),
    businessName: Yup.string().required('Obavezno'),
    businessDescription: Yup.string().required('Obavezno')
})

const theme = createTheme({
    palette:{
        primary:{
            main:"#8fb4d6"
        }
    }
})


const OwnerEdit=()=>{

    const navigate = useNavigate()

    
    const [user, setUser] = useState("")
    
    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [businessName, setBusinessName] = useState("")
    const [businessDescription, setBusinessDescription] = useState("")
    
    const [isError, setError] = useState("");
    const [isErrorVisible, setIsErrorVisible] = useState(false)

    
    useEffect(()=> {
        if (JSON.parse(sessionStorage.getItem('role') as string) === UserRole.ROLE_USER) {
            navigate("../../")
        } else {
            AxiosInstance.get("profile/owner").then(res => {
                setUser(res.data)
                setUsername(res.data.username)
                setBusinessName(res.data.businessName)
                setBusinessDescription(res.data.businessDescription)
            })
        }
    }, []);

    const editOwner = (
        username: string,
        password: string,
        businessName: string,
        businessDescription: string
    ) => {
      return new Promise<AxiosResponse>((resolve, reject) => {
        AxiosInstance.put('/profile/owner/edit', {
          username,
          password,
          businessName,
          businessDescription
        })
          .then(res => {
            resolve(res.data)
          })
          .catch(err => reject(err))
      })
    }

    const formik = useFormik({
        enableReinitialize: true,
        initialValues: {
            username: username,
            password: password,
            passwordConfirmation: password,
            businessName: businessName,
            businessDescription: businessDescription
        },
        validationSchema: EditOwnerSchema,
        onSubmit: values=> {
                editOwner(
                    values.username,
                    values.password,
                    values.businessName,
                    values.businessDescription
                )
                .then(res=>{
                    navigate('/profile/owner')
                })
                .catch(err=>{
                    setIsErrorVisible(true)
                    setError(err.response.data.message);
                })
        }
    })

    return(
        <div className='user-edit'> 

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
                                id="username"
                                name="username" 
                                type="text"
                                onChange={formik.handleChange}
                                value={formik.values.username}/> 
                                ) : (<>
                            <input className='label-input-error'
                                id="username"
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
                            id="password"
                            name="password" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.password}/> 
                            ) : (<>
                        <input className='label-input-error'
                            id="password"
                            name="password" 
                            type="password"
                            onChange={formik.handleChange}
                            value={formik.values.password}
                        /> 
                        <span className='error'>{formik.errors.password}</span> </>)
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
                            id="passwordConfirmation"
                            name="passwordConfirmation" 
                            type="password"
                            onChange={formik.handleChange}
                            /> 
                            ) : (<>
                        <input className='label-input-error'
                            id="passwordConfirmation"
                            name="passwordConfirmation" 
                            type="password"
                            onChange={formik.handleChange}
                            
                        /> 
                        <span className='error'>{formik.errors.passwordConfirmation}</span> </>)
                        }  
                    </div> 
                </div>

                <div className='row2'>
                    <div className='label-name'>
                        <label>
                            Ime obrta:
                        </label>
                    </div>

                    <div className='label-input'>
                        {!(formik.touched.businessName && formik.errors.businessName) ? 
                            (<input className='label'
                                id="businessName"
                                name="businessName" 
                                type="text"
                                onChange={formik.handleChange}
                                value={formik.values.businessName}/> 
                                ) : (<>
                            <input className='label-input-error'
                                id="businessName"
                                name="businessName" 
                                type="text"
                                onChange={formik.handleChange}
                                value={formik.values.businessName}
                            /> 
                            <span className='error'>{formik.errors.businessName}</span> </>)
                        }  
                    </div>  
                </div>

                <div className='row2'>
                    <div className='label-name'>
                        <label>
                            Opis obrta:
                        </label>
                    </div>

                    <div className='label-input'>
                        {!(formik.touched.businessDescription && formik.errors.businessDescription) ? 
                            (<input className='label'
                                id="businessDescription"
                                name="businessDescription" 
                                type="text"
                                onChange={formik.handleChange}
                                value={formik.values.businessDescription}/> 
                                ) : (<>
                            <input className='label-input-error'
                                id="businessDescription"
                                name="businessDescription" 
                                type="text"
                                onChange={formik.handleChange}
                                value={formik.values.businessDescription}
                            /> 
                            <span className='error'>{formik.errors.businessDescription}</span> </>)
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
                            }
                        }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/save.png')}/>Spremi</div></Button>
                    </ThemeProvider>
                    </div>

                    <Link className='button3-edit' to="../profile/owner" style={{ textDecoration: 'none', color: 'black' }}>
                        <div>
                        <ThemeProvider theme={theme}>
                            <Button 
                            color="primary"
                            className='button3' 
                            variant="contained"
                            size="large"
                            sx={{
                                '@media screen and (max-width: 720px)': {
                                    fontSize: 12,
                                }
                            }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/cancel.png')}/>Odustani</div></Button>
                            </ThemeProvider>
                        </div>
                    </Link>
                </div>
                

            </form>
        </div>
    )
}

export default OwnerEdit