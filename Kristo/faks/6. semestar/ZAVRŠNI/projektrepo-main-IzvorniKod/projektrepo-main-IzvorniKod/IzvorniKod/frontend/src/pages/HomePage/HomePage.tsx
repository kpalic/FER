import { Link } from 'react-router-dom'
//styling
import './HomePage.css'

const HomePage=()=>{
    return(
        <>
        <div className='homepage-container'>
            <div className='top-container'>
                <div className='text-container'>
                    <h1 className="headline-dog-friendly">Dog friendly</h1>
                    <div className='description-container'>
                    Pronaći lokacije prikladne i dostojne čovjekovog najboljeg prijatelja
                    nije uvijek lako i jednostavno. 
                    Dog Friendly prikazuje svojim korisnicima, vlasnicima i ljubiteljima
                    pasa korisne lokacije kao što su veterinari, frizerski saloni za pse, ali
                    i lokacije koje nisu prikladne za pse.
                    </div>
                    
                    <div className='button-container'>
                        <Link className='button' to="/map" style={{ textDecoration: 'none', color: 'black' }}>
                            Karta
                        </Link>
                    </div>
                </div>
            </div>
        </div>
        </>
    )
}


export default HomePage;