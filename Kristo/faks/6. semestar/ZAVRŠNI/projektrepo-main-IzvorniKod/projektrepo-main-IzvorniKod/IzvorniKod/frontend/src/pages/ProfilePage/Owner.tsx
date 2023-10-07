import './Owner.css'
import Button from '@mui/material/Button';
import { useState, useEffect } from 'react';
import { AxiosInstance } from '../../auth/AxiosInstance';
import { Link, useNavigate } from 'react-router-dom'
import { ThemeProvider } from '@mui/material/styles'
import { createTheme } from '@mui/material/styles'
import { useAuth } from "../../auth/authContext";
import { AxiosResponse } from 'axios';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { MenuItem, Select } from '@mui/material';
import { UserRole } from "../../models/User.model";

const PromoteBusinessSchema=Yup.object().shape({
    promoteDuration: Yup.string().required('Obavezno')
})

const theme = createTheme({
    palette:{
        primary:{
            main:"#8fb4d6"
        }
    }
})

const Owner=()=>{

    const navigate = useNavigate()

    
    const [owner, setOwner] = useState("")
    const [promotionDuration, setPromotionDuration] = useState("")
    const [promotion, setPromotion] = useState("Vaš obrt nije promoviran")
    
    const [isError, setError] = useState("");
    const [isErrorVisible, setIsErrorVisible] = useState(false)
    
    
    const auth = useAuth()
    
    useEffect(() => {
        if (JSON.parse(sessionStorage.getItem('role') as string) === UserRole.ROLE_USER) {
            navigate("../../")
        } else {
            AxiosInstance.get('profile/owner')
            .then((response) => {
                setOwner(response.data);
                setPromotionDuration(response.data.promotionDuration)
            })
            .catch(err => console.log(err));
        }
    }, []);

    const deleteOwnerProfile = () => {
        AxiosInstance.delete('profile/owner/delete')
        .then((response) => {
            auth.logout()
            navigate('/')
            window.location.reload()
        })
        .catch((err) => {});
    }

    let type;
    if((owner as any).businessType === "SHOP") {
        type = "Trgovina za pse";
    } else if ((owner as any).businessType === "VET") {
        type = "Veterinarska ordinacija"
    } else if ((owner as any).businessType === "DAYCARE") {
        type = "Vrtić za pse";
    } else  if ((owner as any).businessType === "SALON"){
        type = "Salon za uređivanje psa";
    } else if ((owner as any).businessType === "OTHER"){
        type = "Ostalo";
    }


    const promoteBusiness = (
        businessOIB: string,
        promoteDuration: string
    ) => {
      return new Promise<AxiosResponse>((resolve, reject) => {
        AxiosInstance.post('/profile/owner/promote', {
          businessOIB,
          promoteDuration
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
            businessOIB: (owner as any).businessOIB,
            promoteDuration: ''
        },
        validationSchema: PromoteBusinessSchema,
        onSubmit: values=> {
            promoteBusiness(
                    values.businessOIB,
                    values.promoteDuration
                )
                .then(res=> {
                    setPromotionDuration(values.promoteDuration)
                })
                .catch(err=>{
                    setIsErrorVisible(true)
                    setError(err.response.data.message)
                })
        }
    })

    useEffect(() => {
        let dateStart;
        if((owner as any).promotionStart === null) {
            dateStart = new Date();
        } else {
            dateStart = new Date((owner as any).promotionStart)
        }

        if( promotionDuration === "0.25") {
            dateStart.setHours(dateStart.getHours() + 7*24)
        } else {
            dateStart.setMonth(dateStart.getMonth() + parseInt(promotionDuration))
        }

        let now = new Date();

        var diff = (dateStart.getTime() - now.getTime())

        let day = Math.ceil(diff / (1000 * 60 * 60 * 24))
        now.setHours(now.getHours() + 24*day)
        let month = now.getMonth()
        let year = Math.floor(month / 12)

        if(year === 1 || year >= 5) {
            setPromotion(year + " godina");
        } else if (year >= 2 && year < 5) {
            setPromotion(year + " godine")
        } else if (month === 1) {
            setPromotion(month + " mjesec")
        } else if (month >= 2 && month < 5) {
            setPromotion(month + " mjeseca")
        } else if (month >= 5) {
            setPromotion(month + " mjeseci")
        } else if (day === 1) {
            setPromotion(day + " dan")
        } else if (day >= 2) {
            setPromotion(day + " dana")
        } else {
            setPromotion("Vaš obrt nije promoviran.")
        }
    }, [promotionDuration]); 


    return (
        <div className='owner-page'>
            <div className='owner-column'>
                <div className='owner'>
                    <div className='owner-data-title'>
                        <h1>Korisnički podatci </h1>
                </div>

                    <div className='owner-data'>
                        <div className='row'>
                            <div className='data-name'>
                                Korisničko ime:
                            </div>
                            <div className='data'>
                                {(owner as any).username}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Email:
                            </div>
                            <div className='data'>
                                {(owner as any).email}
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
                        <div className='row'>
                            <div className='data-name'>
                                Broj kartice:
                            </div>
                            <div className='data'>
                                {(owner as any).cardNumber + " **** **** ****"} 
                            </div>
                        </div>
                    </div>
                </div>
                <div className='business'>
                    <div className='owner-data-title'>
                        <h1>Podatci o obrtu </h1>
                    </div>
                    <div className='owner-data'>
                        <div className='row'>
                            <div className='data-name'>
                                Ime obrta:
                            </div>
                            <div className='data'>
                                {(owner as any).businessName}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Adresa obrta:
                            </div>
                            <div className='data'>
                                {(owner as any).businessAddress + ", " + (owner as any).businessCity}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Opis obrta:
                            </div>
                            <div className='data'>
                                {(owner as any).businessDescription}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Kontakt-broj obrta:
                            </div>
                            <div className='data'>
                                {(owner as any).businessMobileNumber}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                OIB obrta:
                            </div>
                            <div className='data'>
                                {(owner as any).businessOIB}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Vrsta obrta:
                            </div>
                            <div className='data'>
                                {type}
                            </div>
                        </div>
                        <div className='row'>
                            <div className='data-name'>
                                Trajanje promocije:
                            </div>
                            <div className='data'>
                                {promotion}
                            </div>
                        </div>
                    </div>
                    <div className='buttons-container'>
                        <div className='button-owner-container'>
                            <Link className='button3-edit' to="/profile/owner/edit" style={{ textDecoration: 'none', color: 'black' }}>
                                <ThemeProvider theme={theme}>
                                    <Button
                                    className='button3'
                                    color="primary"
                                    variant="contained"
                                    sx={{
                                        '@media screen and (max-width: 720px)': {
                                            fontSize: 10,
                                        }
                                    }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/edit.png')} alt='edit'/> Uredi podatke</div></Button>
                                </ThemeProvider>        
                            </Link>
                        </div>
                        <div className='button-owner-container'>
                            <ThemeProvider theme={theme}>
                                <Button
                                color="primary"
                                variant="contained"
                                className='button3'
                                onClick={() => {
                                        const confirmBox = window.confirm(
                                            "Želite li zaista izbrisati svoj korisnički račun?"
                                        )
                                        if (confirmBox === true) {
                                            deleteOwnerProfile()
                                        }
                                }}
                                sx={{
                                    '@media screen and (max-width: 720px)': {
                                        fontSize: 10,
                                    }
                                }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/delete2.png')} alt='delete'/>Izbriši korisnički račun</div></Button>
                                </ThemeProvider>
                        </div>
                    </div>
                </div>
            </div>

            <div className='owner-column'>
                <div className='promote-title'>
                        <h1>Promocija obrta </h1>
                </div>
                
                {isErrorVisible ? (<div className='error-container'>{isError}</div> ) : (<></>)}
                <form className='owner-form' onSubmit={formik.handleSubmit}>
                    <div className='promote-form'>
                        <div className='label-name'>
                            <label>
                                Trajanje promocije:
                            </label>
                        </div>
                        
                        <Select
                        id="promoteDuration"
                        name="promoteDuration"
                        label="Vrsta objekta"
                        variant="outlined"
                        disabled={!(promotionDuration === null)}
                        sx={{ marginBottom: '1rem' }}
                        fullWidth={true}
                        value={formik.values.promoteDuration}
                        onChange={formik.handleChange}
                        error={formik.touched.promoteDuration && Boolean(formik.errors.promoteDuration)}
                        
                        >
                        <MenuItem value={0}  disabled>Odaberite trajanje promocije</MenuItem>
                        <MenuItem value={"0.25"}> 1 tjedan (5€) </MenuItem>
                        <MenuItem value={"1"} > 1 mjesec (15€)</MenuItem>
                        <MenuItem value={"2"} > 2 mjeseca (30€)</MenuItem>
                        <MenuItem value={"6"} > 6 mjeseci (90€)</MenuItem>
                        <MenuItem value={"12"} > 1 godina (180€)</MenuItem>
                        <MenuItem value={"24"} > 2 godine (360€)</MenuItem>
                    </Select>  
                    
                    </div>
                    <div className='button-promote-container'>

                        <ThemeProvider theme={theme}>
                            <Button
                            className='button3'
                            color="primary"
                            variant="contained"
                            size="large"
                            disabled={(!(promotionDuration === null)) || (formik.values.promoteDuration === "")}
                            onClick={(e) => {
                                if(!(formik.values.promoteDuration === "0.25")) {
                                    if (window.confirm(
                                        "Jeste li sigurni da želite promovirati svoj obrt za "
                                        + parseInt(formik.values.promoteDuration)*15 +"€ ?") === true) {
    
                                        formik.handleSubmit();
                                    } else {
                                        e.preventDefault();
                                    }
                                } else {
                                    if (window.confirm("Jeste li sigurni da želite promovirati svoj obrt za 5€ ?") === true) 
                                        formik.handleSubmit();
                                }
                            }}
                            sx={{
                                '@media screen and (max-width: 720px)': {
                                    fontSize: 12,
                                }
                            }}><div className='button-content'><img className='button-icon' src={require('./../../assets/icons/promote.png')} alt='promote'/>Promoviraj</div></Button>
                        </ThemeProvider>
                    </div>
                </form>

                <div className='promote-title'>
                    <h1>Kako radi promocija?</h1>
                </div>
                <div className='promote'>
                    Za iznos od 5/15 € po tjednu/mjesecu možete promovirati svoj obrt na našoj stranici.
                    Vaš obrt korisnicima će se dodatno isticati na karti, kao i pod preporučenim obrtima.
                    Trajanje promocije može se produžiti tek nakon isteka trajanja promocije.
                </div>
            </div>
        </div>
    )
}
export default Owner