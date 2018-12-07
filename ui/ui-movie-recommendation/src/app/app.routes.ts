import { Routes } from '@angular/router';
import { IndexComponent } from './index/index.component';


export const routes: Routes = [
  // { path: '', component: IndexComponent  },
  // { path: 'error', component: ErrorComponent },
  // { path: 'login', component: IndexComponent  },
  // { path: 'landing', component: WelcomeComponent, canActivate: [AuthGuard]},
  // { path: 'unauthorized', component: UnauthorizedComponent},
  // { path: 'problemMeters', component: ProblemMetersComponent, canActivate: [AuthGuard], resolve: { user: UserResolver} ,
  // },
  // { path: 'analysisBasket', component: AnalysisBasketCore, canActivate: [AuthGuard], resolve: { user: UserResolver} ,
  // },
  // { path: 'problemMeters/:searchquery', component: ProblemMetersComponent, canActivate: [AuthGuard] },
  // { path: 'metersearch/:searchquery', component: CustomersearchComponent, canActivate: [AuthGuard] },
  // { path: 'meterdetail/:searchquery/:searchvalue', component: CustomerdetailComponent, canActivate: [AuthGuard], resolve: { user: UserResolver}  },
  // { path: 'meterdetail/:searchquery/:searchvalue/:pagename', component: CustomerdetailComponent, canActivate: [AuthGuard], resolve: { user: UserResolver}  },
  // { path: 'meterdetail/:searchquery/:searchvalue/:pagename/:refreshtoken', component: CustomerdetailComponent },
  // { path: 'favoritie', component: FavoritieComponent, canActivate: [AuthGuard] },
  // {
  //   path: 'dashboard', component: DashboardComponent, canActivate: [AuthGuard],
  //   children: [
  //     { path: '', component: MeterReadingComponent },
  //     { path: 'meterReading', component: MeterReadingComponent },
  //     { path: 'meterReadingHealthCard', component: MeterReadingHealthCardComponent },      
  //     { path: 'meterReadingDetails', component: MeterReadingDetailsComponent },
  //     { path: 'meterManagement', component: MeterManagementComponent }]
  //     // { path: 'calendar', component: DashboardCalenderHeatMapComponent }  
  // },
  { path: '**', component: IndexComponent  }
];
