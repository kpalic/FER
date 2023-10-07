package hr.fer.progi.simplicity;


import hr.fer.progi.simplicity.entities.User;
import hr.fer.progi.simplicity.security.CustomUserDetailsService;
import hr.fer.progi.simplicity.services.BusinessService;
import hr.fer.progi.simplicity.services.CardService;
import hr.fer.progi.simplicity.services.ProfileService;
import org.junit.Test;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;
import org.springframework.test.context.junit4.SpringRunner;


import static org.junit.jupiter.api.Assertions.*;

@RunWith(SpringRunner.class)
@SpringBootTest(
        webEnvironment = SpringBootTest.WebEnvironment.MOCK,
        classes = DogFriendlyApplication.class)
@AutoConfigureMockMvc
@TestPropertySource(locations = "classpath:application-unitTest.properties")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class RegistrationUnitTests {
    private static final String username = "user";
    private static final String email = "user.dogfriendly@gmail.com";
    private static final String password = "lozinka123";
    private static final String businessName = "PET_SHOP";
    private static final String businessType = "SHOP";
    private static final String businessAddress = "Zagrebacka 5";
    private static final String businessCity = "Zagreb";
    private static final String businessOIB = "01234567890";
    private static final String businessMobileNumber = "+385 99-234-56-78";
    private static final String businessDescription = "Prodajemo sve za kucne ljubimce";
    private static final String cardNumber = "0000111122223333";
    private static final String expiryDateMonth = "5";
    private static final String expiryDateYear = "2025";
    private static final String cvv3 = "123";
    private static final String cvv4 = "124";

    @Autowired
    private CustomUserDetailsService userService;

    @Autowired
    private ProfileService profileService;

    @Autowired
    private BusinessService businessService;

    @Autowired
    private CardService cardService;

    @Test
    @Order(1)
    public void registrationCheckUsername() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(null, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner("", email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner("?_kriviUsername", email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        //SAME USERNAME
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

    @Test
    @Order(2)
    public void registrationCheckEmail() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(username, null, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, "", password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, "neispravanEmail", password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        //SAME EMAIL
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

    @Test
    @Order(3)
    public void registrationCheckBusinessName() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, null, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, "", businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, "Ovo je preveliko odnosno predugačko ime za postaviti nekom obrtu na našoj stranici", businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        //SAME NAME
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, null, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

    @Test
    @Order(4)
    public void registrationCheckBusinessOIB() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, null, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, "", businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, "0123456789", businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, "012345678901", businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, "0123456789a", businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        //SAME OIB
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

    @Test
    @Order(5)
    public void registrationCheckCardNumber() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, null, expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, "", expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, "0000", expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, "00001111222233334", expiryDateMonth, expiryDateYear, cvv3));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, "aaaabbbbccccdddd", expiryDateMonth, expiryDateYear, cvv3));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

    @Test
    @Order(6)
    public void registrationCheckCVV() {
        //CHECK ALL INFO
        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv3));

        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, null));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, ""));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, "12"));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, "12345"));
        assertThrows(Exception.class, () -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, "abc"));

        User user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());

        assertDoesNotThrow(() -> userService.createOwner(username, email, password, businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription, cardNumber, expiryDateMonth, expiryDateYear, cvv4));

        user = userService.getUserByUsername(username);
        profileService.deleteUserById(user.getId());
    }

}
