import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { RouterModule } from '@angular/router';
import { routes } from './app.routes';
import { IndexComponent } from './index/index.component';
import { HttpClientModule } from '@angular/common/http';
import { CustomersService } from './index/customers.service';

@NgModule({
  declarations: [
    IndexComponent,
    AppComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    RouterModule.forRoot(routes, {
      useHash: true
    })
  ],
  providers: [CustomersService],
  bootstrap: [AppComponent]
})
export class AppModule { }
