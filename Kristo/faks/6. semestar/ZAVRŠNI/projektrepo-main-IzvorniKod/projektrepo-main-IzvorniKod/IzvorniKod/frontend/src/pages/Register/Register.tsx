import { Link } from 'react-router-dom'
import './Register.css'

const Register = () => {
    return(
        <div className='registration'>
            <div>
                <h1 className='registration-title'>
                    Želite li se registrirati kao: 
                </h1>
            </div>

            <div className='role'>
                <div className='column'>
                    <div className='button2-container'>
                        <Link className='button2' to="user" style={{ textDecoration: 'none', color: 'black' }}>
                            <div className='button2-role'>
                                Korisnik
                            </div>
                        </Link>
                    </div>

                    <div className='role-description'>
                        Vlasnik ste čovjekovog najboljeg prijatelja i imate želju dodavati i recenzirati 
                        najpoznatije lokacije za kućne ljubimce? Registrirajte se kao korisnik i otkrijte sve pogodnosti.
                    </div>
                </div>

                <div className='column'>
                    <div className='button2-container'>
                        <Link className='button2' to="owner" style={{ textDecoration: 'none', color: 'black' }}>
                            <div className='button2-role'>
                                Vlasnik obrta
                            </div>
                        </Link>
                    </div>

                    <div className='role-description'>
                        Imate vlastiti obrt povezan s njegom kućnih ljubimaca ili u svom obrtu dopuštate boravak naših najboljih prijatelja? Registrirajte se kao 
                        vlasnik obrta kako biste Vaš obrt prikazali na karti te dobili mogućnost za plaćenu promociju i isticanje.
                    </div>
                </div>
            </div>
        </div>
)}
export default Register;