import "./Footer.css"

const Footer = () => {
    return (
        <div className="main-footer">
            <div className="footer-container">
                <div className="footer-item">
                    <img src='../../../madeBy.svg' alt="Made By Simplicity" className="madeBy"/>
                </div>
                <div className="footer-item">
                    <ul className="footer-list"> Kontaktirajte nas:
                        <li className="footer-list-item">01/6129/999</li>
                        <li className="footer-list-item">simplicity@fer.hr</li>
                        <li className="footer-list-item">Unska 3, Zagreb</li>
                    </ul>
                </div>
            </div>
        </div>
        
    )
}

export default Footer