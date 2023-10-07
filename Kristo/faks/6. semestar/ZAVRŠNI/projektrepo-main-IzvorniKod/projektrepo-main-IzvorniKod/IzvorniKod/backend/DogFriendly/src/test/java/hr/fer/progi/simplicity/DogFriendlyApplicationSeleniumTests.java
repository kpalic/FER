package hr.fer.progi.simplicity;

import static org.junit.Assert.*;


import org.junit.Test;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.runner.RunWith;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.annotation.Order;
import org.springframework.test.context.junit4.SpringRunner;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@RunWith(SpringRunner.class)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@SpringBootTest
public class DogFriendlyApplicationSeleniumTests {

    @Test
    @Order(1)
    public void seleniumWebUserRegistration() {
        WebDriver driver = new ChromeDriver();
        System.setProperty("webdriver.chrome.driver", "C:\\Program Files (x86)\\chromedriver.exe");
        driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
        driver.get("http://localhost:3000/");

        driver.findElement(By.xpath("//a[@href='/auth/login']")).click();

        WebElement element = driver.findElement(By.id("username"));
        element.sendKeys("DF_TestUser");
        element = driver.findElement(By.id("password"));
        element.sendKeys("123");

        driver.findElement(By.xpath("//button[@type='submit']")).click();

        if(driver.findElement(By.className("error-container")).isDisplayed()) System.out.println("Element is Visible");

        driver.findElement(By.xpath("//a[@href='/auth/register']")).click();
        driver.findElement(By.xpath("//a[@href='/auth/register/user']")).click();

        element = driver.findElement(By.id("username"));
        element.sendKeys("DF_TestUser");
        element = driver.findElement(By.id("email"));
        element.sendKeys("user");
        element = driver.findElement(By.id("password"));
        element.sendKeys("123");

        driver.findElement(By.xpath("//button[@type='submit']")).click();

        if(driver.findElement(By.id("email-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("password-helper-text")).isDisplayed()) System.out.println("Element is Visible");

        element = driver.findElement(By.id("email"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("dogfriendly.test.owner1@gmail.com");
        element = driver.findElement(By.id("password"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("12345678");

        driver.findElement(By.xpath("//button[@type='submit']")).click();

        boolean compRes = new WebDriverWait(driver, Duration.ofSeconds(40)).until(ExpectedConditions.urlToBe("http://localhost:3000/auth/login"));

        String redirURL = driver.getCurrentUrl();
        compRes = redirURL.contains("auth/login");
        if (!driver.findElement(By.className("registration-message")).isDisplayed()) compRes = false;
        assertEquals(compRes, true);

        driver.quit();
    }

    @Test
    @Order(2)
    public void seleniumOwnerRegistration() {
        WebDriver driver = new ChromeDriver();
        driver.manage().window().setSize(new Dimension(700, 1300));
        System.setProperty("webdriver.chrome.driver", "C:\\Program Files (x86)\\chromedriver.exe");
        driver.manage().timeouts().implicitlyWait(20, TimeUnit.SECONDS);
        driver.get("http://localhost:3000/");

        driver.findElement(By.className("hamburger")).click();
        driver.findElement(By.className("hamburger")).click();
        driver.findElement(By.className("hamburger")).click();
        driver.findElement(By.xpath("//div[@class='menu-dropdown']/div[3]")).click();

        driver.findElement(By.xpath("//a[@href='/auth/register']")).click();
        driver.findElement(By.xpath("//a[@href='/auth/register/owner']")).click();

        // REGISTRIRAJ SE
        driver.findElement(By.xpath("//button[@type='submit']")).click();

        if(driver.findElement(By.id("username-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("email-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("password-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("businessName-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("businessAdress-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("businessCity-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("businessOIB-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("businessMobileNumber-helper-text")).isDisplayed()) System.out.println("Element is Visible");
        if(driver.findElement(By.id("cardNumber-helper-text")).isDisplayed()) System.out.println("Element is Visible");

        // Upisujemo podatke vlasnika obrta
        WebElement element = driver.findElement(By.id("username"));
        element.sendKeys("DF_TestOwner");
        element = driver.findElement(By.id("email"));
        element.sendKeys("dogfriendly.test.owner@gmail.com");
        element = driver.findElement(By.id("password"));
        element.sendKeys("12345678");
        element = driver.findElement(By.id("businessName"));
        element.sendKeys("TestingBusiness");

        driver.findElement(By.xpath("//form[@class='register-form']/div/div[5]")).click();
        driver.findElement(By.xpath("//li[@data-value='VET']")).click();

        element = driver.findElement(By.id("businessAdress"));
        element.sendKeys("Unska ul. 3");
        element = driver.findElement(By.id("businessCity"));
        element.sendKeys("Zagreb");
        element = driver.findElement(By.id("businessOIB"));
        element.sendKeys("OIB0123456789OIB"); // WARNING
        element = driver.findElement(By.id("businessMobileNumber"));
        element.sendKeys("Broj telefona: +012/3456-789"); // WARNING
        element = driver.findElement(By.id("businessDescription"));
        element.sendKeys("Generic description.");
        element = driver.findElement(By.id("cardNumber"));
        element.sendKeys("Card Number 123456789"); // WARNING
        driver.findElement(By.id("expiryDateMonth")).click();
        driver.findElement(By.xpath("//div[@role='presentation']/div[3]/ul/li[2]")).click();
        driver.findElement(By.id("getExpiryDateProps")).click();
        driver.findElement(By.xpath("//div[@role='presentation']/div[3]/ul/li[4]")).click();
        element = driver.findElement(By.id("cvv"));
        element.sendKeys("1"); // WARNING

        // REGISTRIRAJ SE
        driver.findElement(By.xpath("//button[@type='submit']")).click();

        element = driver.findElement(By.id("businessOIB"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("01234567890");
        element = driver.findElement(By.id("businessMobileNumber"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("+012/3456-789");
        element = driver.findElement(By.id("cardNumber"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("1234567890123456");
        element = driver.findElement(By.id("cvv"));
        element.sendKeys(Keys.CONTROL + "a");
        element.sendKeys(Keys.DELETE);
        element.sendKeys("123");

        // REGISTRIRAJ SE
        driver.findElement(By.xpath("//button[@type='submit']")).click();

        boolean compRes = new WebDriverWait(driver, Duration.ofSeconds(40)).until(ExpectedConditions.urlToBe("http://localhost:3000/auth/login"));

        String redirURL = driver.getCurrentUrl();
        compRes = redirURL.contains("auth/login");
        if (!driver.findElement(By.className("registration-message")).isDisplayed()) compRes = false;
        assertEquals(compRes, true);

        driver.quit();
    }
}